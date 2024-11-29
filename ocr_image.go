package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/cretz/bine/tor"
)

const baseURL = "https://tools.clc.hcmus.edu.vn"
const (
	processedImagesDir = "processed_images"
	responseDir        = "response"
	logFile            = "log.txt"
)

type UploadResponse struct {
	IsSuccess bool        `json:"is_success"`
	Code      string      `json:"code"`
	Message   interface{} `json:"message"`
	Data      UploadData  `json:"data"`
}

type UploadData struct {
	FileName string `json:"file_name"`
}

type OCRResponse struct {
	IsSuccess bool        `json:"is_success"`
	Code      string      `json:"code"`
	Message   interface{} `json:"message"`
	Data      OCRData     `json:"data"`
}

type OCRData struct {
	OCRID          int               `json:"ocr_id,omitempty"`
	OCRName        string            `json:"ocr_name,omitempty"`
	ResultFileName string            `json:"result_file_name,omitempty"`
	ResultOCRText  []string          `json:"result_ocr_text,omitempty"`
	ResultBBox     [][][]interface{} `json:"result_bbox,omitempty"`
}

type ClassificationResponse struct {
	IsSuccess bool               `json:"is_success"`
	Code      string             `json:"code"`
	Message   interface{}        `json:"message"`
	Data      ClassificationData `json:"data"`
}

type ClassificationData struct {
	ClassificationID   int    `json:"ocr_id"`
	ClassificationName string `json:"ocr_name"`
}

type OCRItem struct {
	Text       string       `json:"text"`
	Confidence float64      `json:"confidence"`
	Points     [][2]float64 `json:"points"`
}

var currentIPChangeCount = 0

func main() {
	logFile, err := os.OpenFile(logFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Printf("Failed to open log file: %v\n", err)
		return
	}
	defer logFile.Close()
	logger := log.New(logFile, "", log.LstdFlags)

	err = os.MkdirAll(responseDir, os.ModePerm)
	if err != nil {
		logger.Printf("Failed to create response directory: %v\n", err)
		return
	}

	imageFiles, err := ioutil.ReadDir(processedImagesDir)
	if err != nil {
		logger.Printf("Failed to read processed_images directory: %v\n", err)
		return
	}

	var validImageFiles []os.FileInfo
	for _, file := range imageFiles {
		if !file.IsDir() && isImageFile(file.Name()) {
			validImageFiles = append(validImageFiles, file)
		}
	}

	batchSize := 2
	totalBatches := (len(validImageFiles) + batchSize - 1) / batchSize

	var torProxy *tor.Tor
	var client *http.Client

	dataDirs, err := getDataDirs()
	if err != nil || len(dataDirs) == 0 {
		fmt.Println("No data directories found.")
		return
	}

	dataDirIndex := 0

	for batch := 0; batch < totalBatches; batch++ {
		if batch%2 == 0 {
			if torProxy != nil {
				torProxy.Close()
			}
			dataDir := dataDirs[dataDirIndex%len(dataDirs)]
			dataDirIndex++

			torProxy, client, err = switchIP(dataDir)
			if err != nil {
				fmt.Printf("Error switching IP: %v\n", err)
				continue
			}
		}

		start := batch * batchSize
		end := start + batchSize
		if end > len(validImageFiles) {
			end = len(validImageFiles)
		}
		currentBatch := validImageFiles[start:end]

		var wg sync.WaitGroup
		for _, file := range currentBatch {
			wg.Add(1)
			go func(file os.FileInfo) {
				defer wg.Done()
				imagePath := filepath.Join(processedImagesDir, file.Name())
				err := processImage(imagePath, client)
				if err != nil {
					logger.Printf("Failed to process image %s: %v", file.Name(), err)
				} else {
					logger.Printf("Successfully processed image %s", file.Name())
				}
			}(file)
		}

		wg.Wait()
	}

	if torProxy != nil {
		torProxy.Close()
	}
}

func getDataDirs() ([]string, error) {
	var dirs []string
	currentDir, err := filepath.Abs(".")
	if err != nil {
		return nil, err
	}
	files, err := ioutil.ReadDir(currentDir)
	if err != nil {
		return nil, err
	}
	for _, f := range files {
		if f.IsDir() && strings.HasPrefix(f.Name(), "data-dir-") {
			absPath := filepath.Join(currentDir, f.Name())
			dirs = append(dirs, absPath)
		}
	}
	return dirs, nil
}

func switchIP(dataDir string) (*tor.Tor, *http.Client, error) {
	currentIPChangeCount++
	fmt.Printf("Changing IP... (%d) using data directory: %s\n", currentIPChangeCount, dataDir)

	torProxy, err := startTor(dataDir)
	if err != nil {
		fmt.Printf("Error starting Tor: %v\n", err)
		return nil, nil, err
	}

	client := getTorClient(torProxy)

	fmt.Println("Switched IP successfully!")
	return torProxy, client, nil
}

func startTor(dataDir string) (*tor.Tor, error) {
	t, err := tor.Start(nil, &tor.StartConf{DataDir: dataDir})
	if err != nil {
		return nil, fmt.Errorf("could not start Tor with dataDir %s: %w", dataDir, err)
	}
	return t, nil
}

func getTorClient(t *tor.Tor) *http.Client {
	dialCtx, err := t.Dialer(context.Background(), nil)
	if err != nil {
		fmt.Printf("Error getting Tor dialer: %v\n", err)
		return nil
	}
	client := &http.Client{
		Transport: &http.Transport{
			DialContext: dialCtx.DialContext,
		},
	}
	return client
}

func isImageFile(filename string) bool {
	extensions := []string{".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
	ext := strings.ToLower(filepath.Ext(filename))
	for _, e := range extensions {
		if ext == e {
			return true
		}
	}
	return false
}

func processImage(imagePath string, client *http.Client) error {
	imageName := filepath.Base(imagePath)
	fileName, err := uploadImage(imagePath, client)
	if err != nil {
		return fmt.Errorf("failed to upload image: %w", err)
	}

	var forceClassID int
	forceClassID = 1 // Comment if want to classify image

	if forceClassID == 0 {
		classificationData, err := classifyImage(fileName, client)
		if err != nil {
			return fmt.Errorf("classification error: %w", err)
		}
		forceClassID = classificationData.ClassificationID
	}

	ocrData, err := ocrImage(fileName, forceClassID, client)
	if err != nil {
		return fmt.Errorf("OCR error: %w", err)
	}

	err = saveOCRResults(imageName, ocrData)
	if err != nil {
		return fmt.Errorf("failed to save OCR results: %w", err)
	}

	err = os.Remove(imagePath)
	if err != nil {
		return fmt.Errorf("failed to delete image %s: %w", imagePath, err)
	}

	return nil
}

func uploadImage(imagePath string, client *http.Client) (string, error) {
	url := fmt.Sprintf("%s/api/web/clc-sinonom/image-upload", baseURL)

	file, err := os.Open(imagePath)
	if err != nil {
		return "", fmt.Errorf("unable to open image file: %w", err)
	}
	defer file.Close()

	var requestBody bytes.Buffer
	writer := multipart.NewWriter(&requestBody)
	part, err := writer.CreateFormFile("image_file", filepath.Base(imagePath))
	if err != nil {
		return "", fmt.Errorf("failed to create form file: %w", err)
	}
	_, err = io.Copy(part, file)
	if err != nil {
		return "", fmt.Errorf("failed to copy file data: %w", err)
	}
	err = writer.Close()
	if err != nil {
		return "", fmt.Errorf("failed to close writer: %w", err)
	}

	req, err := http.NewRequest("POST", url, &requestBody)
	if err != nil {
		return "", fmt.Errorf("failed to create POST request: %w", err)
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("POST request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response body: %w", err)
	}

	var uploadResp UploadResponse
	err = json.Unmarshal(body, &uploadResp)
	if err != nil {
		return "", fmt.Errorf("failed to parse JSON response: %w", err)
	}
	if !uploadResp.IsSuccess {
		return "", fmt.Errorf("upload failed with code %s", uploadResp.Code)
	}
	return uploadResp.Data.FileName, nil
}

func classifyImage(fileName string, client *http.Client) (ClassificationData, error) {
	url := fmt.Sprintf("%s/api/web/clc-sinonom/image-classification", baseURL)

	bodyMap := map[string]string{"file_name": fileName}
	bodyBytes, err := json.Marshal(bodyMap)
	if err != nil {
		return ClassificationData{}, fmt.Errorf("failed to marshal JSON: %w", err)
	}

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(bodyBytes))
	if err != nil {
		return ClassificationData{}, fmt.Errorf("failed to create POST request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return ClassificationData{}, fmt.Errorf("POST request failed: %w", err)
	}
	defer resp.Body.Close()

	responseBody, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return ClassificationData{}, fmt.Errorf("failed to read response body: %w", err)
	}

	var classificationResp ClassificationResponse
	err = json.Unmarshal(responseBody, &classificationResp)
	if err != nil {
		return ClassificationData{}, fmt.Errorf("failed to parse JSON response: %w", err)
	}
	if !classificationResp.IsSuccess {
		return ClassificationData{}, fmt.Errorf("classification failed with code %s", classificationResp.Code)
	}
	return classificationResp.Data, nil
}

func ocrImage(fileName string, ocrID int, client *http.Client) (OCRData, error) {
	url := fmt.Sprintf("%s/api/web/clc-sinonom/image-ocr", baseURL)

	bodyMap := map[string]string{
		"file_name": fileName,
		"ocr_id":    fmt.Sprintf("%d", ocrID),
	}
	bodyBytes, err := json.Marshal(bodyMap)
	if err != nil {
		return OCRData{}, fmt.Errorf("failed to marshal JSON: %w", err)
	}

	maxRetries := 3
	retryDelay := time.Second * 1

	for attempt := 1; attempt <= maxRetries; attempt++ {
		req, err := http.NewRequest("POST", url, bytes.NewBuffer(bodyBytes))
		if err != nil {
			return OCRData{}, fmt.Errorf("failed to create POST request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := client.Do(req)
		if err != nil {
			if attempt < maxRetries {
				time.Sleep(retryDelay)
				retryDelay *= 2
				continue
			}
			return OCRData{}, fmt.Errorf("POST request failed after %d attempts: %w", maxRetries, err)
		}
		defer resp.Body.Close()

		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			body, _ := ioutil.ReadAll(resp.Body)
			if attempt < maxRetries {
				time.Sleep(retryDelay)
				retryDelay *= 2
				continue
			}
			return OCRData{}, fmt.Errorf("server returned error: %d %s, body: %s", resp.StatusCode, resp.Status, string(body))
		}

		responseBody, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			return OCRData{}, fmt.Errorf("failed to read response body: %w", err)
		}

		var ocrResp OCRResponse
		err = json.Unmarshal(responseBody, &ocrResp)
		if err != nil {
			if attempt < maxRetries {
				time.Sleep(retryDelay)
				retryDelay *= 2
				continue
			}
			return OCRData{}, fmt.Errorf("failed to parse JSON response after %d attempts: %w", maxRetries, err)
		}

		if !ocrResp.IsSuccess {
			if attempt < maxRetries {
				time.Sleep(retryDelay)
				retryDelay *= 2
				continue
			}
			return OCRData{}, fmt.Errorf("OCR failed with code %s", ocrResp.Code)
		}

		return ocrResp.Data, nil
	}

	return OCRData{}, fmt.Errorf("OCR failed after %d attempts", maxRetries)
}

func saveOCRResults(imageName string, ocrData OCRData) error {
	baseName := strings.TrimSuffix(imageName, filepath.Ext(imageName))
	txtFileName := fmt.Sprintf("%s.txt", baseName)
	txtFilePath := filepath.Join(responseDir, txtFileName)

	var ocrItems []OCRItem

	for i, bboxItem := range ocrData.ResultBBox {
		if i >= len(ocrData.ResultOCRText) {
			log.Printf("Mismatch between ResultOCRText and ResultBBox for image %s at index %d", imageName, i)
			break
		}

		if len(bboxItem) < 2 {
			log.Printf("Invalid bboxItem format for image %s at index %d", imageName, i)
			continue
		}

		pointsRaw := bboxItem[0]

		var points [][2]float64
		for _, pointRaw := range pointsRaw {
			point, ok := pointRaw.([]interface{})
			if !ok || len(point) != 2 {
				log.Printf("Invalid point pair for image %s at index %d", imageName, i)
				continue
			}

			x, xOk := point[0].(float64)
			y, yOk := point[1].(float64)
			if !xOk || !yOk {
				log.Printf("Invalid coordinate values for image %s at index %d", imageName, i)
				continue
			}

			points = append(points, [2]float64{x, y})
		}

		textInfo := bboxItem[1]

		text, textOk := textInfo[0].(string)
		confidence, confOk := textInfo[1].(float64)
		if !textOk || !confOk {
			log.Printf("Invalid text or confidence format for image %s at index %d", imageName, i)
			continue
		}

		ocrItems = append(ocrItems, OCRItem{
			Text:       text,
			Confidence: confidence,
			Points:     points,
		})
	}

	ocrJSON, err := json.MarshalIndent(ocrItems, "", "    ")
	if err != nil {
		return fmt.Errorf("failed to marshal OCR items to JSON: %v", err)
	}

	finalContent := fmt.Sprintf("%s %s\n", imageName, string(ocrJSON))

	err = ioutil.WriteFile(txtFilePath, []byte(finalContent), 0644)
	if err != nil {
		return fmt.Errorf("failed to write to txt file: %v", err)
	}

	return nil
}

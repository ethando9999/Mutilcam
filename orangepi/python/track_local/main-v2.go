// main.go (PHIÊN BẢN FRAME-BASED - TÔN TRỌNG TOTAL_DETECT)
package main

import (
	"context"
	"encoding/json"
	"log"
	"math"
	"math/big"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"go-server-smc/blockchain"
	"go-server-smc/datacollector"
	"go-server-smc/personanalytics"

	"github.com/ethereum/go-ethereum/core/types"
	"github.com/gorilla/websocket"
)

// --- STRUCTS & CONSTANTS ---
type ColorInfoEvent struct {
	RGB        []uint8 `json:"rgb"`
	Percentage float64 `json:"percentage"`
}
type DetectionEvent struct {
	FrameID      int64            `json:"frame_id"`
	TotalDetect  int              `json:"total_detect"`
	PersonID     string           `json:"person_id"`
	Gender       string           `json:"gender"`
	TorsoColor   []ColorInfoEvent `json:"torso_color"`
	TorsoStatus  string           `json:"torso_status"`
	PantsColor   []ColorInfoEvent `json:"pants_color"`
	PantsStatus  string           `json:"pants_status"`
	SkinTone     []uint8          `json:"skin_tone"`
	Height       *float64         `json:"height"`
	TimeDetect   string           `json:"time_detect"`
	CameraID     string           `json:"camera_id"`
	WorldPointXY []float64        `json:"world_point_xy"`
}
type AnalyticsRequest struct {
	PersonID    string
	Gender      string
	TorsoColors []personanalytics.PersonAnalyticsColorInfo
	TorsoStatus string
	PantsColors []personanalytics.PersonAnalyticsColorInfo
	PantsStatus string
	SkinTone    personanalytics.PersonAnalyticsColorRGB
	Height      *big.Int
	TimeDetect  *big.Int
	CameraID    string
	FrameID     *big.Int
	WorldPointX *big.Int
	WorldPointY *big.Int
}
type TransactionRequest struct {
	IsDataCollector bool
	DC_CameraID     string
	DC_FrameID      *big.Int
	DC_TimeDetect   *big.Int
	DC_Detections   []datacollector.CameraDataCollectorFullStorageDetectedObject
	PA_Request      AnalyticsRequest
}
type FrameData struct {
	TotalDetect int
	Detections  []DetectionEvent
	ReceivedAt  time.Time
}

var (
	transactionQueue = make(chan TransactionRequest, 200)
	pendingFrames    = make(map[int64]*FrameData)
	frameMutex       sync.Mutex
	upgrader         = websocket.Upgrader{CheckOrigin: func(r *http.Request) bool { return true }}
)

const (
	FRAME_TIMEOUT          = 5 * time.Second
	FRAME_CLEANUP_INTERVAL = 10 * time.Second
	COORDINATE_MULTIPLIER  = 100
	X_COORDINATE_OFFSET    = 150.0
	HEIGHT_MULTIPLIER      = 10000.0
	MAX_COLORS_TO_STORE    = 7
)

// --- GOROUTINES ---
func transactionDispatcher() {
	log.Println("✅ [Dispatcher] Sẵn sàng xử lý cả 2 loại giao dịch.")
	auth := blockchain.Auth
	currentNonce, err := blockchain.Client.PendingNonceAt(context.Background(), auth.From)
	if err != nil {
		log.Fatalf("Lỗi lấy nonce: %v", err)
	}
	log.Printf("   - Nonce khởi tạo là: %d", currentNonce)

	for req := range transactionQueue {
		var tx *types.Transaction
		var txErr error
		for i := 0; i < 3; i++ {
			pendingNonce, _ := blockchain.Client.PendingNonceAt(context.Background(), auth.From)
			if pendingNonce > currentNonce {
				log.Printf("⚠️ [Dispatcher] Đồng bộ Nonce! Cũ: %d, Mới: %d", currentNonce, pendingNonce)
				currentNonce = pendingNonce
			}
			auth.Nonce = big.NewInt(int64(currentNonce))

			if req.IsDataCollector {
				log.Printf("--- [Dispatcher] Bắt đầu xử lý TX cho DataCollector (Frame %d) ---", req.DC_FrameID)
				tx, txErr = blockchain.DataCollectorContractInstance.RecordCameraData(auth, req.DC_CameraID, req.DC_FrameID, req.DC_TimeDetect, req.DC_Detections)
			} else {
				paReq := req.PA_Request
				log.Printf("--- [Dispatcher] Bắt đầu xử lý TX cho PersonAnalytics (ID %s...) ---", paReq.PersonID[:8])
				tx, txErr = blockchain.PersonAnalyticsContractInstance.UpdateProfile(auth, paReq.PersonID, paReq.Gender, paReq.TorsoColors, paReq.TorsoStatus, paReq.PantsColors, paReq.PantsStatus, paReq.SkinTone, paReq.Height, paReq.TimeDetect, paReq.CameraID, paReq.FrameID, paReq.WorldPointX, paReq.WorldPointY)
			}
			if txErr != nil {
				log.Printf("[Dispatcher] [Lần thử %d] LỖI gửi TX: %v", i+1, txErr)
				if strings.Contains(strings.ToLower(txErr.Error()), "nonce") {
					time.Sleep(1 * time.Second)
					continue
				}
				break
			}
			log.Printf("   ✔ Giao dịch ĐÃ ĐƯỢC GỬI! Hash: %s", tx.Hash().Hex())
			currentNonce++
			break
		}
	}
}

func frameCleanupManager() {
	ticker := time.NewTicker(FRAME_CLEANUP_INTERVAL)
	defer ticker.Stop()
	log.Printf("✅ [Frame Manager] Sẵn sàng dọn dẹp các frame không hoàn chỉnh mỗi %v.", FRAME_CLEANUP_INTERVAL)

	for range ticker.C {
		frameMutex.Lock()
		now := time.Now()
		for frameID, frame := range pendingFrames {
			if now.Sub(frame.ReceivedAt) > FRAME_TIMEOUT {
				log.Printf("   [Frame Manager] Frame #%d đã timeout (nhận %d/%d). Đang xóa...", frameID, len(frame.Detections), frame.TotalDetect)
				delete(pendingFrames, frameID)
			}
		}
		frameMutex.Unlock()
	}
}

func handleConnections(w http.ResponseWriter, r *http.Request) {
	ws, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("Lỗi nâng cấp kết nối: %v", err)
		return
	}
	defer ws.Close()
	log.Printf("✅ Client %s đã kết nối.", ws.RemoteAddr())

	for {
		_, messageBytes, err := ws.ReadMessage()
		if err != nil {
			break
		}

		var event DetectionEvent
		if err := json.Unmarshal(messageBytes, &event); err != nil {
			log.Printf("Lỗi giải mã JSON: %v. Payload: %s", err, string(messageBytes))
			continue
		}

		if event.PersonID == "" {
			continue
		}

		log.Printf("Nhận được dữ liệu cho PersonID %s... thuộc Frame #%d (Total: %d)", event.PersonID[:8], event.FrameID, event.TotalDetect)

		frameMutex.Lock()
		frame, exists := pendingFrames[event.FrameID]
		if !exists {
			if event.TotalDetect <= 0 {
				log.Printf("   - Bỏ qua frame mới #%d vì total_detect không hợp lệ: %d", event.FrameID, event.TotalDetect)
				frameMutex.Unlock()
				continue
			}
			log.Printf("   - Frame mới #%d. Cần nhận %d người.", event.FrameID, event.TotalDetect)
			frame = &FrameData{
				TotalDetect: event.TotalDetect,
				Detections:  make([]DetectionEvent, 0, event.TotalDetect),
				ReceivedAt:  time.Now(),
			}
			pendingFrames[event.FrameID] = frame
		}

		frame.Detections = append(frame.Detections, event)
		log.Printf("   - Frame #%d đã nhận %d/%d người.", event.FrameID, len(frame.Detections), frame.TotalDetect)

		if len(frame.Detections) >= frame.TotalDetect {
			log.Printf("   ✔ Frame #%d đã HOÀN CHỈNH. Chuẩn bị gửi giao dịch.", event.FrameID)
			go processCompleteFrame(*frame)
			delete(pendingFrames, event.FrameID)
		}
		frameMutex.Unlock()
	}
}

func processCompleteFrame(frame FrameData) {
	if len(frame.Detections) == 0 {
		return
	}

	// Lấy thông tin chung từ người đầu tiên
	firstPerson := frame.Detections[0]
	cameraID := firstPerson.CameraID
	frameID := firstPerson.FrameID

	const customTimeLayout = "2006-01-02T15:04:05.999999"
	t, err := time.Parse(customTimeLayout, firstPerson.TimeDetect)
	if err != nil {
		log.Printf("Lỗi parse thời gian trong processCompleteFrame: %v", err)
		return
	}
	timestamp := big.NewInt(t.Unix())

	// --- 1. Chuẩn bị cho DataCollector ---
	var dcDetections []datacollector.CameraDataCollectorFullStorageDetectedObject
	for _, person := range frame.Detections {
		transformedX := person.WorldPointXY[0] + X_COORDINATE_OFFSET
		pointY := person.WorldPointXY[1]
		dcDetections = append(dcDetections, datacollector.CameraDataCollectorFullStorageDetectedObject{
			PersonId:    person.PersonID,
			WorldPointX: big.NewInt(int64(math.Round(transformedX * COORDINATE_MULTIPLIER))),
			WorldPointY: big.NewInt(int64(math.Round(pointY * COORDINATE_MULTIPLIER))),
			WorldPointZ: big.NewInt(0),
		})
	}

	dcRequest := TransactionRequest{
		IsDataCollector: true,
		DC_CameraID:     cameraID,
		DC_FrameID:      big.NewInt(frameID),
		DC_TimeDetect:   timestamp,
		DC_Detections:   dcDetections,
	}
	transactionQueue <- dcRequest
	log.Printf("   ✔ Đã thêm yêu cầu cho DataCollector (Frame #%d) vào hàng đợi.", frameID)

	// --- 2. Chuẩn bị cho PersonAnalytics (cho từng người trong frame) ---
	for _, personEvent := range frame.Detections {
		prepareAndQueueAnalyticsUpdate(personEvent)
	}
}

func prepareAndQueueAnalyticsUpdate(event DetectionEvent) {
	if len(event.WorldPointXY) < 2 {
		return
	}

	var torsoColors []personanalytics.PersonAnalyticsColorInfo
	if len(event.TorsoColor) > MAX_COLORS_TO_STORE {
		event.TorsoColor = event.TorsoColor[:MAX_COLORS_TO_STORE]
	}
	for _, c := range event.TorsoColor {
		if len(c.RGB) == 3 {
			torsoColors = append(torsoColors, personanalytics.PersonAnalyticsColorInfo{R: c.RGB[0], G: c.RGB[1], B: c.RGB[2], Percentage: uint8(c.Percentage)})
		}
	}

	var pantsColors []personanalytics.PersonAnalyticsColorInfo
	if len(event.PantsColor) > MAX_COLORS_TO_STORE {
		event.PantsColor = event.PantsColor[:MAX_COLORS_TO_STORE]
	}
	for _, c := range event.PantsColor {
		if len(c.RGB) == 3 {
			pantsColors = append(pantsColors, personanalytics.PersonAnalyticsColorInfo{R: c.RGB[0], G: c.RGB[1], B: c.RGB[2], Percentage: uint8(c.Percentage)})
		}
	}

	var skinTone personanalytics.PersonAnalyticsColorRGB
	if len(event.SkinTone) == 3 {
		skinTone = personanalytics.PersonAnalyticsColorRGB{R: event.SkinTone[0], G: event.SkinTone[1], B: event.SkinTone[2]}
	}

	var height int64 = 0
	if event.Height != nil {
		height = int64(*event.Height * HEIGHT_MULTIPLIER)
	}

	const customTimeLayout = "2006-01-02T15:04:05.999999"
	t, _ := time.Parse(customTimeLayout, event.TimeDetect)

	paRequest := AnalyticsRequest{
		PersonID:    event.PersonID,
		Gender:      event.Gender,
		TorsoColors: torsoColors,
		TorsoStatus: event.TorsoStatus,
		PantsColors: pantsColors,
		PantsStatus: event.PantsStatus,
		SkinTone:    skinTone,
		Height:      big.NewInt(height),
		TimeDetect:  big.NewInt(t.Unix()),
		CameraID:    event.CameraID,
		FrameID:     big.NewInt(event.FrameID),
		WorldPointX: big.NewInt(int64(event.WorldPointXY[0])),
		WorldPointY: big.NewInt(int64(event.WorldPointXY[1])),
	}
	transactionQueue <- TransactionRequest{IsDataCollector: false, PA_Request: paRequest}
	log.Printf("   ✔ Đã thêm yêu cầu cho PersonAnalytics (ID %s...) vào hàng đợi.", event.PersonID[:8])
}

func main() {
	blockchain.Init()
	go transactionDispatcher()
	go frameCleanupManager()

	http.HandleFunc("/api/ws/camera", handleConnections)
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	log.Printf("🚀 Server (Frame-based) đang lắng nghe trên cổng %s...", port)
	if err := http.ListenAndServe(":"+port, nil); err != nil {
		log.Fatalf("Lỗi: %v", err)
	}
}

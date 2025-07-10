package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"log"
	"net"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// --- Cấu hình ---
const (
	SERVER_PORT           = 5050
	BUFFER_SIZE           = 1500
	DEPTH_DIR             = "depth_frames"
	AMPLITUDE_DIR         = "amplitude_frames"
	CONFIDENCE_DIR        = "confidence_frames" // <-- THÊM MỚI
	FRAME_TYPE_DEPTH      = 0
	FRAME_TYPE_AMPLITUDE  = 1
	FRAME_TYPE_CONFIDENCE = 2 // <-- THÊM MỚI
	CLEANUP_INTERVAL      = 5 * time.Second
)

type FrameKey struct {
	ID   uint32
	Type uint8
}

type FrameBuffer struct {
	sync.Mutex
	Chunks      map[uint32][]byte
	TotalChunks uint32
	LastSeen    time.Time
}

var frameReassemblyBuffer = &sync.Map{}

// processPacket: Giữ nguyên logic từ phiên bản gốc đã hoạt động
func processPacket(data []byte, remoteAddr net.Addr) {
	if len(data) < 13 {
		log.Printf("Packet from %s too short: %d bytes", remoteAddr.String(), len(data))
		return
	}

	frameID := binary.BigEndian.Uint32(data[0:4])
	frameType := data[4]
	totalChunks := binary.BigEndian.Uint32(data[5:9])
	chunkIndex := binary.BigEndian.Uint32(data[9:13])

	if chunkIndex >= totalChunks {
		return
	}

	key := FrameKey{ID: frameID, Type: frameType}
	val, _ := frameReassemblyBuffer.LoadOrStore(key, &FrameBuffer{
		Chunks:      make(map[uint32][]byte),
		TotalChunks: totalChunks,
	})

	buffer := val.(*FrameBuffer)
	buffer.Lock()
	defer buffer.Unlock()

	if _, exists := buffer.Chunks[chunkIndex]; exists {
		return
	}
	buffer.Chunks[chunkIndex] = data[13:]
	buffer.LastSeen = time.Now()

	if len(buffer.Chunks) == int(buffer.TotalChunks) {
		log.Printf("Frame %d (Type: %d) fully assembled with %d chunks.", key.ID, key.Type, buffer.TotalChunks)
		chunksToProcess := make(map[uint32][]byte, len(buffer.Chunks))
		for k, v := range buffer.Chunks {
			chunksToProcess[k] = v
		}
		frameReassemblyBuffer.Delete(key)
		go assembleAndSave(key, chunksToProcess, buffer.TotalChunks)
	}
}

// assembleAndSave: Giữ nguyên logic
func assembleAndSave(key FrameKey, chunks map[uint32][]byte, totalChunks uint32) {
	var frameData bytes.Buffer
	for i := uint32(0); i < totalChunks; i++ {
		chunk, ok := chunks[i]
		if !ok {
			log.Printf("Error: Missing chunk %d for frame %d (Type: %d).", i, key.ID, key.Type)
			return
		}
		frameData.Write(chunk)
	}
	saveFrameToFile(key, frameData.Bytes())
}

// saveFrameToFile: Nâng cấp để xử lý confidence
func saveFrameToFile(key FrameKey, data []byte) {
	var dir string
	var typeStr string
	switch key.Type {
	case FRAME_TYPE_DEPTH:
		dir = DEPTH_DIR
		typeStr = "Depth"
	case FRAME_TYPE_AMPLITUDE:
		dir = AMPLITUDE_DIR
		typeStr = "Amplitude"
	case FRAME_TYPE_CONFIDENCE: // <-- NÂNG CẤP
		dir = CONFIDENCE_DIR
		typeStr = "Confidence"
	default:
		log.Printf("Unknown frame type: %d for frame ID %d", key.Type, key.ID)
		return
	}

	filePath := filepath.Join(dir, fmt.Sprintf("frame_%d.bin", key.ID))
	err := os.WriteFile(filePath, data, 0644)
	if err != nil {
		log.Printf("Error saving frame %d to %s: %v", key.ID, filePath, err)
	} else {
		// Log chi tiết hơn để dễ theo dõi
		log.Printf("Saved %s Frame %d to %s (%d bytes)", typeStr, key.ID, filePath, len(data))
	}
}

// cleanupStaleFrames: Giữ nguyên logic
func cleanupStaleFrames() {
    ticker := time.NewTicker(CLEANUP_INTERVAL)
    defer ticker.Stop()
    for range ticker.C {
        now := time.Now()
        frameReassemblyBuffer.Range(func(key, value interface{}) bool {
            buffer := value.(*FrameBuffer)
            buffer.Lock()
            if buffer != nil && now.Sub(buffer.LastSeen) > CLEANUP_INTERVAL {
                frameReassemblyBuffer.Delete(key)
            }
            buffer.Unlock()
            return true
        })
    }
}


func main() {
	log.Println("Starting UDP Receiver v2.2 (Upgraded from Stable)...")

	// NÂNG CẤP: Tạo cả 3 thư mục
	for _, dir := range []string{DEPTH_DIR, AMPLITUDE_DIR, CONFIDENCE_DIR} {
		if err := os.MkdirAll(dir, 0755); err != nil {
			log.Fatalf("Failed to create directory %s: %v", dir, err)
		}
	}

	// go cleanupStaleFrames() // Tắt để debug

	addr, err := net.ResolveUDPAddr("udp", fmt.Sprintf("0.0.0.0:%d", SERVER_PORT))
	if err != nil {
		log.Fatalf("Failed to resolve address: %v", err)
	}

	conn, err := net.ListenUDP("udp", addr)
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}
	defer conn.Close()
	
	conn.SetReadBuffer(2 * 1024 * 1024)
	log.Printf("Receiver ready, listening on port %d", SERVER_PORT)

	// Giữ nguyên vòng lặp main từ phiên bản gốc đã hoạt động
	buffer := make([]byte, BUFFER_SIZE)
	for {
		n, remoteAddr, err := conn.ReadFromUDP(buffer)
		if err != nil {
			log.Printf("Error reading UDP: %v", err)
			continue
		}
		packetData := make([]byte, n)
		copy(packetData, buffer[:n])
		go processPacket(packetData, remoteAddr)
	}
}

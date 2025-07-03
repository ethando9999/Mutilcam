package main

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"log"
	"net"
	"strconv"
	"strings"
	"sync"
	"time"

	// "io/ioutil"
	"math/rand"
	"os"
	"os/exec"
	"runtime"
)

const (
	SERVER_PORT = 5050
	PASSWORD    = "secret_password"
	INIT_FLAG   = 1
	END_FLAG    = 2
	ERROR_FLAG  = 3
)

var (
	frameMap   = make(map[string]*Frame)
	frameMapMu sync.Mutex
	sentFrames = 0
	startTime  = time.Now()
	unixSocket string
)

var AESKey []byte

func init() {
	hash := sha256.Sum256([]byte(PASSWORD))
	AESKey = hash[:]
}

type Frame struct {
	UUID             string
	TotalChunks      int
	ChunkSize        int
	ReceivedData     map[int][]byte
	ReceivedChecksum map[int]int
	EndChecksum      uint16      // Chỉ cần checksum để biết gói END đã đến
	TimeoutTimer     *time.Timer // Bộ đếm thời gian để kiểm tra timeout
	mu               sync.Mutex
}

// Kiểm tra xem đã có RAMDisk nào đang được mount trên macOS hay không
func checkMacRamDisk() bool {
	// Sử dụng lệnh "mount" để lấy thông tin các ổ đã được mount
	out, err := exec.Command("mount").Output()
	if err != nil {
		fmt.Println("Lỗi khi lấy thông tin mount:", err)
		return false
	}

	// Kiểm tra xem có dòng nào chứa "/Volumes/RAMDisk"
	if strings.Contains(string(out), "/Volumes/RAMDisk") {
		return true
	}
	return false
}

// Kiểm tra xem đã có RAMDisk nào đang được mount trên Linux hay không
func checkLinuxRamDisk() bool {
	// Kiểm tra trong /proc/mounts hoặc sử dụng lệnh "mount"
	out, err := exec.Command("mount").Output()
	if err != nil {
		fmt.Println("Lỗi khi lấy thông tin mount:", err)
		return false
	}

	// Kiểm tra xem có dòng nào chứa "tmpfs" (thường là kiểu của RAMDisk)
	if strings.Contains(string(out), "tmpfs") && strings.Contains(string(out), "/ramdisk") {
		return true
	}
	return false
}

// Kiểm tra sự tồn tại của Unix Socket file
func checkUnixSocket() bool {
	// Kiểm tra xem Unix socket có tồn tại không
	_, err := os.Stat(unixSocket)
	fmt.Printf("unixSocket", unixSocket)
	return !os.IsNotExist(err)
}

func unpadPKCS7(data []byte) ([]byte, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("invalid padding: empty data")
	}

	padding := int(data[len(data)-1])
	if padding == 0 || padding > len(data) {
		return nil, fmt.Errorf("invalid padding")
	}

	for _, v := range data[len(data)-padding:] {
		if int(v) != padding {
			return nil, fmt.Errorf("invalid padding")
		}
	}

	return data[:len(data)-padding], nil
}

func decryptAES(data []byte) ([]byte, error) {
	// Kiểm tra độ dài dữ liệu
	if len(data) < aes.BlockSize {
		return nil, fmt.Errorf("invalid ciphertext: too short")
	}

	// Tách IV và ciphertext
	iv := data[:aes.BlockSize]
	ciphertext := data[aes.BlockSize:]
	if len(ciphertext)%aes.BlockSize != 0 {
		return nil, fmt.Errorf("invalid ciphertext: not a multiple of block size")
	}

	// Giải mã AES-CBC
	block, err := aes.NewCipher(AESKey)
	if err != nil {
		return nil, fmt.Errorf("failed to create AES cipher: %v", err)
	}

	mode := cipher.NewCBCDecrypter(block, iv)
	mode.CryptBlocks(ciphertext, ciphertext)

	// Loại bỏ padding
	return unpadPKCS7(ciphertext)
}

func calculateChecksum(frameNumber byte, sequence uint16, data []byte, uuid string) uint16 {
	// Chuẩn bị payload
	payload := append([]byte{frameNumber}, byte(sequence>>8), byte(sequence))
	payload = append(payload, data...)
	payload = append(payload, []byte(uuid)...)
	payload = append(payload, []byte(PASSWORD)...)

	// Tính toán checksum và giới hạn trong 2 byte
	return uint16(crc32.ChecksumIEEE(payload) % 65536)
}

func handleInit(frameNumber byte, sequenceNumber uint16, checksum uint16, encryptedPayload []byte, addr *net.UDPAddr, conn *net.UDPConn) error {

	combinedData := append([]byte{INIT_FLAG}, encryptedPayload...)

	// Giải mã payload
	decrypted, err := decryptAES(encryptedPayload)
	if err != nil {
		return fmt.Errorf("failed to decrypt init payload: %v", err)
	}

	// Kiểm tra payload hợp lệ
	if len(decrypted) < 10 {
		return fmt.Errorf("invalid decrypted INIT payload")
	}

	// Tách các giá trị từ payload
	uuid := string(decrypted[:6])
	chunkSize := binary.BigEndian.Uint16(decrypted[6:8])
	totalChunks := binary.BigEndian.Uint16(decrypted[8:10])

	// Log thông tin frame
	// log.Printf("handleInit --- Frame Number: %d, UUID: %s, Chunk Size: %d, Total Chunks: %d", frameNumber, uuid, chunkSize, totalChunks)

	// Kiểm tra checksum
	calculatedChecksum := calculateChecksum(frameNumber, sequenceNumber, combinedData, uuid)
	if calculatedChecksum != checksum {
		// return fmt.Errorf("checksum mismatch for sequence %d with uuid: %s", sequenceNumber, uuid)
		return fmt.Errorf("handleInit - checksum mismatch for sequence %d with uuid: %s, received: %d, expected: %d", sequenceNumber, uuid, calculatedChecksum, checksum)
	}

	// Lưu thông tin frame vào map
	frameMapMu.Lock()
	defer frameMapMu.Unlock()

	frameMap[string(frameNumber)] = &Frame{
		UUID:             uuid,
		TotalChunks:      int(totalChunks),
		ChunkSize:        int(chunkSize),
		ReceivedData:     make(map[int][]byte),
		ReceivedChecksum: make(map[int]int),
	}

	// Phản hồi "OK" tới client
	_, err = conn.WriteTo([]byte("OK"), addr)
	// fmt.Printf("Successfully handle init for frameNumber %d\n", frameNumber)
	return err
}

func handleChunk(frameNumber byte, sequenceNumber uint16, checksum uint16, payload []byte, addr *net.UDPAddr, conn *net.UDPConn) error {
	// Kiểm tra độ dài payload (tối thiểu 128 bytes mã hóa)
	if len(payload) < 160 {
		return fmt.Errorf("invalid chunk payload: too short")
	}

	// Lấy thông tin frame từ map
	frameMapMu.Lock()
	frame, exists := frameMap[string(frameNumber)]
	frameMapMu.Unlock()
	if !exists {
		// return fmt.Errorf("frame not initialized for frameNumber: %s", string(frameNumber))
		return fmt.Errorf("frame not initialized for frameNumber: %d", frameNumber)
	}

	// Lưu dữ liệu chunk
	frame.mu.Lock()
	defer frame.mu.Unlock()

	if _, exists := frame.ReceivedChecksum[int(sequenceNumber)]; exists {
		return fmt.Errorf("handleChunk - duplicate package for sequence %d with frameNumber: %s", sequenceNumber, frameNumber)
	}

	uuid := frame.UUID

	// Kiểm tra checksum
	calculatedChecksum := calculateChecksum(frameNumber, sequenceNumber, payload, uuid)
	if calculatedChecksum != checksum {
		return fmt.Errorf("handleChunk - checksum mismatch for sequence %d with uuid: %s, received: %d, expected: %d", sequenceNumber, uuid, calculatedChecksum, checksum)
	}

	var combinedData []byte
	if sequenceNumber == 1 || sequenceNumber == 10 { // Nếu là chunk được mã hóa AES

		// Tách phần mã hóa AES (128 bytes đầu + 32 byte iv) và phần không mã hóa
		encryptedPart := payload[:160]
		plainPart := payload[160:]

		decryptedPart, err := decryptAES(encryptedPart) // Giải mã phần mã hóa
		if err != nil {
			return fmt.Errorf("failed to decrypt chunk payload: %v", err)
		}
		combinedData = append(decryptedPart, plainPart...) // Kết hợp phần giải mã và phần không mã hóa
	} else {
		combinedData = payload // Giữ nguyên dữ liệu cho các chunk không mã hóa
	}

	// Tách UUID từ phần đầu của dữ liệu kết hợp
	if len(combinedData) < 6 {
		return fmt.Errorf("invalid chunk combined data: missing UUID")
	}

	frame.ReceivedData[int(sequenceNumber)] = combinedData
	frame.ReceivedChecksum[int(sequenceNumber)] = int(checksum)

	// Nếu EndChecksum đã tồn tại và đã nhận đủ chunk, xử lý ngay
	if frame.EndChecksum != 0 && len(frame.ReceivedChecksum) == frame.TotalChunks {
		if frame.TimeoutTimer != nil {
			frame.TimeoutTimer.Stop()
			frame.TimeoutTimer = nil
		}
		err := processEndFrame(frameNumber, frame, uuid, addr, conn)
		return err
	}

	// log.Printf("Chunk received: Frame Number: %d, Sequence Number: %d, UUID: %s", frameNumber, sequenceNumber, uuid)
	return nil
}

func handleEnd(frameNumber byte, sequenceNumber uint16, checksum uint16, encryptedPayload []byte, addr *net.UDPAddr, conn *net.UDPConn) error {

	// Giải mã payload
	decrypted, err := decryptAES(encryptedPayload)
	if err != nil {
		return fmt.Errorf("failed to decrypt end payload: %v", err)
	}

	// Kiểm tra payload hợp lệ
	if len(decrypted) < 12 {
		return fmt.Errorf("invalid decrypted END payload")
	}

	// Tách các giá trị từ payload
	uuid := string(decrypted[:6])
	frameSize := binary.BigEndian.Uint32(decrypted[6:10])
	totalChecksum := binary.BigEndian.Uint16(decrypted[10:12])

	// Lấy thông tin frame từ map
	frameMapMu.Lock()
	frame, exists := frameMap[string(frameNumber)]
	frameMapMu.Unlock()
	if !exists {
		return fmt.Errorf("frame not initialized for UUID: %s", uuid)
	}

	// Xác minh các chunk đã nhận
	frame.mu.Lock()
	defer frame.mu.Unlock()

	if string(uuid) != string(frame.UUID) {
		return fmt.Errorf("frame uuid mismatch for UUID: %s and %s", uuid, frame.UUID)
	}

	totalChunks := (int(frameSize) + frame.ChunkSize - 1) / frame.ChunkSize

	if frame.TotalChunks != totalChunks {
		return fmt.Errorf("frame size mismatch for UUID: %s", uuid)
	}

	if frame.TimeoutTimer != nil {
		return fmt.Errorf("frame timeout has init. Ignore...")
	}

	frame.EndChecksum = totalChecksum

	if len(frame.ReceivedChecksum) != frame.TotalChunks {
		frame.TimeoutTimer = time.AfterFunc(1*time.Second, func() {
			frame.mu.Lock()
			defer frame.mu.Unlock()

			frame.TimeoutTimer = nil
			if len(frame.ReceivedChecksum) != frame.TotalChunks {
				missingChunks := []string{}
				for i := 1; i <= frame.TotalChunks; i++ {
					if _, ok := frame.ReceivedChecksum[i]; !ok {
						missingChunks = append(missingChunks, strconv.Itoa(i))
					}
				}
				if len(missingChunks) > 0 {
					conn.WriteTo([]byte("FAILED:"+strings.Join(missingChunks, ",")), addr)
					log.Printf("Failed Frame Number: %d", frameNumber)
				}
				// frameMapMu.Lock()
				// delete(frameMap, string(frameNumber))
				// frameMapMu.Unlock()
			}
		})
		return nil
	}

	return processEndFrame(frameNumber, frame, uuid, addr, conn)
}

func processEndFrame(frameNumber byte, frame *Frame, uuid string, addr *net.UDPAddr, conn *net.UDPConn) error {

	if len(frame.ReceivedChecksum) != frame.TotalChunks {
		return fmt.Errorf("checksum not equal for UUID: %s received: %d, expected: %d", uuid, len(frame.ReceivedChecksum), frame.TotalChunks)
	}

	// Xác minh kích thước frame
	calculatedChecksum := 0
	for i := 1; i <= frame.TotalChunks; i++ {
		calculatedChecksum = (calculatedChecksum + frame.ReceivedChecksum[i]) % 65536
	}

	// calculatedChecksum = binary.BigEndian.Uint16(calculatedChecksum)
	if uint16(calculatedChecksum) != frame.EndChecksum {
		return fmt.Errorf("checksum mismatch for UUID: %s received: %d, expected: %d", uuid, calculatedChecksum, frame.EndChecksum)
	}

	// ham bất đồng bộ cho việc lưu file
	var buffer bytes.Buffer
	for i := 1; i <= frame.TotalChunks; i++ {
		if chunk, ok := frame.ReceivedData[i]; ok {
			buffer.Write(chunk)
		} else {
			return fmt.Errorf("ReceivedData mismatch for UUID: %s index: %d", uuid, i)
		}
	}

	// Gửi dữ liệu cho server Python (ví dụ: Unix socket).
	sendToProcessor(buffer.Bytes())

	frameMapMu.Lock()
	delete(frameMap, string(frameNumber))
	frameMapMu.Unlock()

	// Phản hồi "SUCCESS" tới client
	_, err := conn.WriteTo([]byte("SUCCESS"), addr)
	// log.Printf("Successfully received FrameNumber: %d", frameNumber)

	// Thống kê FPS.
	sentFrames++
	elapsed := time.Since(startTime)
	if elapsed.Seconds() >= 1 {
		fps := float64(sentFrames) / elapsed.Seconds()
		fmt.Printf("FPS: %.2f\n", fps)
		sentFrames = 0
		startTime = time.Now()
	}
	return err
}

func handlePacket(data []byte, addr *net.UDPAddr, conn *net.UDPConn) {
	if len(data) < 7 { // Header: frame_number (1) + sequence_number (2) + checksum (2)
		log.Printf("Packet too short, length: %d", len(data))
		return
	}

	// Đọc header
	frameNumber := data[0]
	sequenceNumber := binary.BigEndian.Uint16(data[1:3])
	checksum := binary.BigEndian.Uint16(data[3:5])

	// Xử lý phần payload (bỏ header)
	payload := data[5:]

	if sequenceNumber > 0 {
		if err := handleChunk(frameNumber, sequenceNumber, checksum, payload, addr, conn); err != nil {
			log.Printf("Error handling chunk in frameNumber %d: %v", frameNumber, err)
		}
		return
	}

	// Đọc flag từ payload
	flag := payload[0]
	encryptedPayload := payload[1:]

	switch flag {
	case INIT_FLAG:
		if err := handleInit(frameNumber, sequenceNumber, checksum, encryptedPayload, addr, conn); err != nil {
			log.Printf("Error handling init: %v", err)
		}
	case END_FLAG:
		// log.Printf("Handling END_FLAG")
		if err := handleEnd(frameNumber, sequenceNumber, checksum, encryptedPayload, addr, conn); err != nil {
			log.Printf("Error handling end: %v", err)
		}
	default:
		log.Printf("Error not recognize flag: %d", flag)
	}

}

func GenerateRandomString(n int) string {
	const letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" 
	result := make([]byte, n)
	for i := range result {
		result[i] = letters[rand.Intn(len(letters))]
	}
	return string(result)
}

// sendToProcessor gửi frame đã xử lý tới Processor qua Unix socket.
func sendToProcessor(frame []byte) {
	// #debug
	// Write content directly to a file
	// timestamp := time.Now().UnixNano()
	// filename := fmt.Sprintf("./tmp/frame_%d.txt", timestamp)
	// timestamp := time.Now().Unix()
	// randomStr := GenerateRandomString(6) // 6-character random string
	// filename := fmt.Sprintf("./output/go/frame_%d_%s.webp", timestamp, randomStr)
	// err := ioutil.WriteFile(filename, frame, 0644) // 0644 is the file permission
	// if err != nil {
	// 	fmt.Println("Error writing file:", err)
	// 	return
	// }

	// return
	conn, err := net.Dial("unix", unixSocket)
	if err != nil {
		fmt.Println("Error connecting to Processor socket:", err)
		return
	}
	defer conn.Close()

	_, err = conn.Write(frame)
	if err != nil {
		fmt.Println("Error sending frame to Processor:", err)
	}
}

func main() {

	systemPlatform := runtime.GOOS

	var ramDiskMounted bool

	if systemPlatform == "darwin" { // macOS
		ramDiskMounted = checkMacRamDisk()
		unixSocket = "/Volumes/RAMDisk/ai_socket"
	} else if systemPlatform == "linux" || systemPlatform == "linux2" { // Linux hoặc Raspberry Pi
		ramDiskMounted = checkLinuxRamDisk()
		// unixSocket      = "/ramdisk/ai_socket"
		unixSocket = "/mnt/ramdisk/ai_socket"
	} else {
		fmt.Println("Hệ điều hành không hỗ trợ %s.", systemPlatform)
		return
	}

	if !ramDiskMounted {
		fmt.Println("Không có RAMDisk đã được mount.")
		return
	}

	if !checkUnixSocket() {
		fmt.Println("Không có unixSocket được cài .")
		return
	}

	addr := net.UDPAddr{
		Port: SERVER_PORT,
		IP:   net.ParseIP("192.168.7.1"),
	}
	conn, err := net.ListenUDP("udp", &addr)
	if err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
	defer conn.Close()

	// conn.SetReadBuffer(4 * 1024 * 1024) // Tăng buffer lên 4MB

	log.Printf("Server started on %s", addr.String())

	buffer := make([]byte, 8000)
	for {
		n, clientAddr, err := conn.ReadFromUDP(buffer)
		if err != nil {
			log.Printf("Error reading from UDP: %v", err)
			continue
		}

		if n < 7 { // Header: frame_number (1) + sequence_number (2) + checksum (2)
			log.Printf("main packet too short, length: %d", n)
			return
		}

		// Tạo slice mới và sao chép dữ liệu
		packet := make([]byte, n)
		copy(packet, buffer[:n])

		go handlePacket(packet, clientAddr, conn)
	}
}

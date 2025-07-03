package main

import (
	"fmt"
	"orangepi/golang/server"
)

var SERVER_PORT = []int{5050, 5051, 5052, 5053} // Danh sách các cổng server

func main() {
	ramdisks := []string{"/mnt/ramdisk1", "/mnt/ramdisk2", "/mnt/ramdisk3", "/mnt/ramdisk4"} // Danh sách ramdisk

	// Bỏ ai_socket
	// Tạo danh sách unixSocket
	var unixSockets []string
	for i := 1; i <= len(ramdisks); i++ { // Chỉ sử dụng ramdisks
		unixSockets = append(unixSockets, fmt.Sprintf("%s/socket", ramdisks[i-1])) // Tạo unixSocket cho từng ramdisk
	}

	// Khởi động server cho từng cổng trong SERVER_PORT
	for i, port := range SERVER_PORT {
		if i >= len(ramdisks) || i >= len(unixSockets) { // Kiểm tra chỉ số
			break // Thoát khỏi vòng lặp nếu chỉ số vượt quá
		}

		go server.StartServer(ramdisks[i], unixSockets[i], port) // Gọi hàm start_server cho từng cổng
	}

	// Giữ cho chương trình chạy
	select {}
}

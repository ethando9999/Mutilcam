# CamPC with Raspberry PI

## How to start 
### init server certificate and sever key:
```
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout server.key -out server.crt -subj "/C=VN/ST=Hanoi/L=Hanoi/O=MyOrganization/OU=IT/CN=center.pi/emailAddress=admin@center.pi"
```

### Adjust the fake domain center.pi to point to the server ip:
```
sudo nano /etc/hosts
```
- Add text in hosts
```
[ipv4] center.pi
```

### Configure tmpfs for Unix Socket
- make dir
```
sudo mkdir -p /mnt/ramdisk
```
```
sudo nano /etc/fstab
```
- add text in fstab
```
tmpfs /mnt/ramdisk tmpfs defaults,size=16M 0 0
```
- Check tmpfs is working properly
```
df -h | grep ramdisk
```

## Setup rsync to automatically upload the file to the Raspberry Pi every time the file changes
###  install inotify-tools:
```
sudo apt update
sudo apt install inotify-tools
```
### Grant execution permission to the script:
```
chmod +x auto_rsync.sh
```
### Set up SSH Key to avoid having to enter your password every time.
- gen ssh:
```
ssh-keygen -t rsa -b 4096
```
- copy ssh to Raspberry Pi:
```
ssh-copy-id pi@192.168.1.248
```
- text:
```
ssh pi@192.168.1.248
```
### Run:
-  Give execute permission to the file
```
chmod +x auto_rsync.sh
```
```
./auto_rsync.sh
```

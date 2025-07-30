#!/usr/bin/sudo /bin/bash
sudo modprobe -r iwlwifi mac80211
sudo modprobe iwlwifi connector_log=0x1

sudo service network-manager stop
SLEEP_TIME=2
WLAN_INTERFACE=$1
if [ "$#" -ne 3 ]; then
	echo "Going to use default settings!"
	chn=64
	bw=HT20
else
	chn=$2
	bw=$3
fi
echo "Bringing $WLAN_INTERFACE down......"
ifconfig $WLANINTERFACE down
while [ $? -ne 0 ]
do
	ifconfig $WLAN_INTERFACE down
done
sleep $SLEEP_TIME
echo "Set $WLAN_INTERFACE into monitor mode......"
iwconfig $WLAN_INTERFACE mode monitor
while [ $? -ne 0 ]
do
	iwconfig $WLAN_INTERFACE mode monitor
done
sleep $SLEEP_TIME
echo "Bringing $WLAN_INTERFACE up......"
ifconfig $WLAN_INTERFACE up
while [ $? -ne 0 ]
do
	ifconfig $WLAN_INTERFACE up
done
sleep $SLEEP_TIME
echo "Set channel $chn $bw..."
iw $WLAN_INTERFACE set channel $chn $bw

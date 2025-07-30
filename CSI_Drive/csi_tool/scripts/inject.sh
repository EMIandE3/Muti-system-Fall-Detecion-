#!/usr/bin/sudo /bin/bash
sudo service network-manager stop
WLAN_INTERFACE=$1
SLEEP_TIME=2
modprobe iwlwifi debug=0x40000
if [ "$#" -ne 3 ]; then
    echo "Going to use default settings!"
    chn=64
    bw=HT20
else
    chn=$2
    bw=$3
fi
sleep $SLEEP_TIME
ifconfig $WLAN_INTERFACE 2>/dev/null 1>/dev/null
while [ $? -ne 0 ]
do
    ifconfig $WLAN_INTERFACE 2>/dev/null 1>/dev/null
done
sleep $SLEEP_TIME
echo "Add monitor mon0....."
iw dev $WLAN_INTERFACE interface add mon0 type monitor
sleep $SLEEP_TIME
echo "Bringing $WLAN_INTERFACE down....."
ifconfig $WLAN_INTERFACE down
while [ $? -ne 0 ]
do
    ifconfig $WLAN_INTERFACE down
done
sleep $SLEEP_TIME
echo "Bringing mon0 up....."
ifconfig mon0 up
while [ $? -ne 0 ]
do
    ifconfig mon0 up
done
sleep $SLEEP_TIME
echo "Set channel $chn $bw....."
iw mon0 set channel $chn $bw



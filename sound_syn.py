### Boot the SuperCollier server first before running sound_synth()!!!
from pythonosc import udp_client
import time

def sound_synth():
    # Setup the OSC client
    client = udp_client.SimpleUDPClient("127.0.0.1", 57120)

    # Frequency values to test
    midinote = [82, 79, 77, 79, 84, 81, 79, 81, 0] #世上只有妈妈好

    # Send each frequency value with a delay
    for midi in midinote:
        print(f"Sending Midinote Number: {midi} ")
        client.send_message("/from_python", midi)
        time.sleep(2)  # Wait for 2 seconds before sending the next frequency

sound_synth()

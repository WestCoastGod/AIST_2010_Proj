### Boot the SuperCollier server first before running sound_synth()!!!
from pythonosc import udp_client
import time

def sound_synth():
    # Setup the OSC client
    client = udp_client.SimpleUDPClient("127.0.0.1", 57120)
    
    midi_list = [81, 79, 76, 79, 84, 81, 79, 81, 76, 79, 81, 79, 76, 74, 72, 79,76, 74, 74, 74, 76, 79, 79, 81, 76, 74, 72, 79, 76, 74, 72, 69, 72, 67, 0]
    # Send each frequency value with a delay
    
    for midinote in midi_list:
        print(f"Sending Midinote Number: {midinote} ")
        client.send_message("/from_python", midinote)
        time.sleep(1)

sound_synth()

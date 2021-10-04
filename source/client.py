"""Small example OSC client

This program sends 10 random values between 0.0 and 1.0 to the /filter address,
waiting for 1 seconds between each value.
"""
import argparse
import random
import time
from pythonosc import udp_client

client = udp_client.SimpleUDPClient("127.0.0.1", 5005)


def send(word):
    client.send_message("/coglione", word)

def rec(status):
    client.send_message("/rec", status)

def meter(status):
    client.send_message("/meter", status)

def meter2():
    for i in range(2000):
        client.send_message("/meter", 1)
        time.sleep(0.2)

def culo(word):
    client.send_message("/culo", word)

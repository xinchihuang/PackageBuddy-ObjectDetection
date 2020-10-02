import rclpy
from rclpy.node import Node

import time
from std_msgs.msg import String
#from ODdemo.msg import Movement

from tkinter import *
#import tkMessageBox
#import tkSimpleDialog
import math, numpy
import _thread
import socket
import select
import struct
import random
import sys, glob # for listing serial ports
import os  # to command the mp3 and wav player omxplayer

try:
    import serial
except ImportError:
    print ("Import error.  Please install pyserial.")
    raise

global START
START = True

connection = None
global FAILURE
FAILURE = False

def toTwosComplement2Bytes( value ):
        """ returns two bytes (ints) in high, low order
        whose bits form the input value when interpreted in
        two's complement
        """
        # if positive or zero, it's OK
        if value >= 0:
            eqBitVal = value
            # if it's negative, I think it is this
        else:
            eqBitVal = (1<<16) + value
    
        return ( (eqBitVal >> 8) & 0xFF, eqBitVal & 0xFF )

# sendCommandASCII takes a string of whitespace-separated, ASCII-encoded base 10 values to send
def sendCommandASCII(command):
    """
    cmd = ""
    for v in command.split():
        cmd += chr(int(v))
    sendCommandRaw(cmd)
    """
    res=b""
    for v in command.split():
        res+=struct.pack(">B",int(v))
    sendCommandRaw(res)

# sendCommandRaw takes a string interpreted as a byte array
def sendCommandRaw(command):
    global connection
    try:
        if connection is not None:
            connection.write(command)
        else:
            print( "Not connected.")
    except serial.SerialException:
        print ("Lost connection")
        connection = None
    #print ' '.join([ str(ord(c)) for c in command ])

# getDecodedBytes returns a n-byte value decoded using a format string.
# Whether it blocks is based on how the connection was set up.
def getDecodedBytes( n, fmt):
    global connection
        
    try:
        return struct.unpack(fmt, connection.read(n))[0]
    except serial.SerialException:
        print ("Lost connection")
        tkMessageBox.showinfo('Uh-oh', "Lost connection to the robot!")
        connection = None
        return None
    except struct.error:
        print ("Got unexpected data from serial port.")
        return None

def bytesOfR( r ):
        """ for looking at the raw bytes of a sensor reply, r """
        print('raw r is', r)
        for i in range(len(r)):
            print('byte', i, 'is', ord(r[i]))
        print('finished with formatR')

def toBinary( val, numBits ):
        """ prints numBits digits of val in binary """
        if numBits == 0:  return
        toBinary( val>>1 , numBits-1 )
#        print((val & 0x01), end=' ')  # print least significant bit

def bitOfByte( bit, byte ):
    """ returns a 0 or 1: the value of the 'bit' of 'byte' """
    if bit < 0 or bit > 7:
        print('Your bit of', bit, 'is out of range (0-7)')
        print('returning 0')
        return 0
    return ((byte >> bit) & 0x01)

# get8Unsigned returns an 8-bit unsigned value.
def get8Unsigned():
    return getDecodedBytes(1, "B")

# get lowest bit from an unsigned byte
def getLowestBit():
    wheelsAndBumpsByte = getDecodedBytes(1, "B")
    print( wheelsAndBumpsByte)
    return bitOfByte(0, wheelsAndBumpsByte)

# get second lowest bit from an unsigned byte
def getSecondLowestBit():
    wheelsAndBumpsByte = getDecodedBytes(1, "B")
    print( wheelsAndBumpsByte)
    return bitOfByte(1, wheelsAndBumpsByte)

def bumped():
    sendCommandASCII('142 7') 
    time.sleep( 0.02 )
    bumpedByte = getDecodedBytes( 1, "B" )
    if bumpedByte == 0:
        return False
    elif bumpedByte > 3:
        print ("CRAZY BUMPER SIGNAL!")
    else:
        return True

def cleanButtonPressed():
    sendCommandASCII('142 18') 
    buttonByte = getDecodedBytes( 1, "B" )
    if buttonByte == 0:
        return False
    elif buttonByte == 1:
        print ("Clean Button Pressed!")
        return True
    elif buttonByte == 4:
        return False
    else:
        print ("Some other button pressed!")
        FAILURE = True
        return False

def dockButtonPressed():
    sendCommandASCII('142 18') 
    buttonByte = getDecodedBytes( 1, "B" )
#    if buttonByte <> 4:
    if buttonByte!=4:
        return False
    else:
        print( "Dock button pressed!")
        return True

def shudder( period, magnitude, numberOfShudders):
    i = 0
    timestep = 0.02
    while i < numberOfShudders:
        i = i + 1
	#shake left
        t = 0
        while t < period:
            driveDirectRot( 0, magnitude )
            t = t + timestep
            time.sleep( timestep )
	#Shake right
        t = 0
        while t < period:
            driveDirectRot( 0, -magnitude )
            t = t + timestep
            time.sleep( timestep )
    driveDirect( 0, 0 )  # stop the previous motion command

def onConnect():
    global connection

    if connection is not None:
        print( "Oops- You're already connected!")
        return

    try:
        ports = getSerialPorts()
        print( "Available ports:\n" + '   '.join(ports))
        #port = raw_input("Port? Enter COM port to open.\nAvailable options:\n" + '\n'.join(ports))
        port = str(ports[0])  # I'm guessing that the Roomba port is first in the list.  So far this works!  :)
    except EnvironmentError:
        port = raw_input("Port?  Enter COM port to open.")

    if port is not None:
        print ("Trying " + str( port ) + "... ")
    try:   #:tty
        #connection = serial.Serial('/dev/ttyUSB0', baudrate=115200, timeout=1)
        #connection = serial.Serial( str(port), baudrate=115200, timeout=1 )
        connection = serial.Serial( str(ports[0]), baudrate=115200, timeout=1 )
        print ("Connected!")
    except:
        print ("Failed.  Could not connect to " + str( port ))

def getSerialPorts():
    """Lists serial ports
    From http://stackoverflow.com/questions/12090503/listing-available-com-ports-with-python

    :raises EnvironmentError:
        On unsupported or unknown platforms
    :returns:
        A list of available serial ports
    """
    if sys.platform.startswith('win'):
        ports = ['COM' + str(i + 1) for i in range(256)]

    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this is to exclude your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')

    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')

    else:
        raise EnvironmentError('Unsupported platform')
    
    #print(ports)
    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result    

def driveDirectTime( left, right, duration ):
    print("driveDirectTime()")
    t = 0   # initialize timer
    while t < duration:
        driveDirect( left, right )
        time.sleep( 0.05 )
        t = t + .05
    driveDirect( 0, 0 )  # stop

def driveDirect( leftCmSec = 0, rightCmSec = 0 ):
    """ sends velocities of each wheel independently
           left_cm_sec:  left  wheel velocity in cm/sec (capped at +- 50)
           right_cm_sec: right wheel velocity in cm/sec (capped at +- 50)
    """
    print( "driveDirect()")
    if leftCmSec < -50: leftCmSec = -50
    if leftCmSec > 50:  leftCmSec = 50
    if rightCmSec < -50: rightCmSec = -50
    if rightCmSec > 50: rightCmSec = 50
    # convert to mm/sec, ensure we have integers
    leftHighVal, leftLowVal = toTwosComplement2Bytes( int( leftCmSec * 10 ) )
    rightHighVal, rightLowVal = toTwosComplement2Bytes( int( rightCmSec * 10 ) )

    # send these bytes and set the stored velocities
    byteListRight = ( rightHighVal , rightLowVal )
    byteListLeft = ( leftHighVal , leftLowVal )
    sendCommandRaw(struct.pack( ">Bhh", 145, int(rightCmSec * 10), int(leftCmSec * 10) ))
    return

def driveDirectRot( robotCmSec = 0, rotation = 0 ):
    """ implements the driveDirect with a given rotation
        Positive rotation turns the robot CCW
        Negative rotation turns the robot CW
    """
    print ("driveDirectRot()")
    vl = robotCmSec - rotation/2
    vr = robotCmSec + rotation/2
    driveDirect ( vl, vr )

def initiateRobotCommunication():
    print( "Initiating Communications to the Create 2 Robot...")
    onConnect()
    time.sleep( 0.3 )
    sendCommandASCII('128')   # Start Open Interface in Passive
    time.sleep( 0.3 )
    sendCommandASCII('140 3 1 64 16 141 3')  # Beep
    time.sleep( 0.3 )
    #sendCommandASCII('131')   # Safe mode
    sendCommandASCII( '132' )   # Full mode 
    time.sleep( 0.3 )
    sendCommandASCII('140 3 1 64 16 141 3')  # Beep
    time.sleep( 0.1 )
    sendCommandASCII('139 4 0 255')  # Turn on Clean and Dock buttons
    time.sleep( 0.03 )

def closeRobotCommunication():
    print( "Closing Communication to the Create 2 Robot...")
    driveDirect( 0, 0 )  # stop robot if moving
    time.sleep( 0.05 )
    sendCommandASCII('140 3 1 64 16 141 3')  # Beep
    time.sleep( 0.3 )
    #sendCommandASCII('139 0 0 0')  # Turn off Clean and Dock buttons
    time.sleep( 0.03 )
    sendCommandASCII('138 0')  # turn off vacuum, etractors, and side brush
    time.sleep( 0.03 )
    #sendCommandASCII( '7' )  # Resets the robot 	
    sendCommandASCII( '173' )  # Stops the Open Interface to Roomba
    time.sleep( 0.3 )
    connection.close()
    time.sleep( 0.1 )
    raise SystemExit	#  Exit program

class Driver(Node):


    def __init__(self):
        super().__init__('driver_node')
        self.publisher_ = self.create_publisher(String, 'status', 10)
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.subscription = self.create_subscription(
            String,
            'OD_topic',
            self.listener_callback,
            10)
        self.subscription

    def timer_callback(self):
        msg = String()
        if START:
            msg.data = '1 1'
            self.publisher_.publish(msg)
        else:
            msg.data = '0 0'
            self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)

    def listener_callback(self, msg):
        if (msg.data == "Door Detected"):
            START = False
            msg = String()
            msg.data = '0 0'
            self.publisher_.publish(msg)
            self.get_logger().info('Listener Publishing: "%s"' % msg.data)
            driveDirect(0,0)
            time.sleep(2)
            driveDirectRot( 0, -10)
            time.sleep(3)
            driveDirect(0,0)
            time.sleep(2)
            driveDirectRot(0,10)
            time.sleep(3)
            driveDirect(5,5)
            START = True
            msg.data = '1 1'
            self.publisher_.publish(msg)
            self.get_logger().info('Listener Publishing: "%s"' % msg.data)

def main(args=None):
    initiateRobotCommunication()
    START = True
    rclpy.init(args=args)
    driver_object = Driver()
    driveDirect(5,5)
    rclpy.spin(driver_object)


    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    driver_object.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

#include <Servo.h>

Servo myServo;  // Create a servo object
int servoPin = 9;  // Pin connected to the servo

void setup() {
    Serial.begin(9600); 
     // Start serial communication at 9600 bps
    myServo.attach(servoPin); 
    myServo.write(90); // Attach the servo to the specified pin
}

void loop() {
    // Check if data is available to read from the serial port
    if (Serial.available() > 0) {
        String command = Serial.readStringUntil('\n');  // Read the command until newline

        // Control the servo based on the command received
        if (command == "biodegradable") {
           
             
            myServo.write(90);
            delay(1000);
           myServo.write(130);
           delay(1000);
            myServo.write(90);
            delay(1000);
                // Wait for a second to allow the servo to reach the position
        } 
        else if (command == "non_biodegradable") {
            
            myServo.write(90);
            delay(1000);
           myServo.write(15);
           delay(1000);
            myServo.write(90);
            delay(1000);   // Wait for a second to allow the servo to reach the position
        }
    }
}

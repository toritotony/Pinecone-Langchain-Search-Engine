#!/bin/bash

# Include the commands from start.sh
# Assuming start.sh is executable
./start.sh

# Include the commands from startup.txt
# Assuming startup.txt contains the command to start your application, e.g., Gunicorn
source startup.txt

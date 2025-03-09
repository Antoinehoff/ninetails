#!/bin/bash

# Set the package name
PACKAGE_NAME="ninetails"

# Remove the build directory if it exists
if [ -d "/home/ah1032/ninetails/build" ]; then
    echo "Removing build directory..."
    rm -rf /home/ah1032/ninetails/build
    rm -rf /home/ah1032/ninetails/ninetails.egg-info
fi

# Reinstall the package
echo "Installing $PACKAGE_NAME..."
pip install /home/ah1032/ninetails > /home/ah1032/ninetails/install.log

echo "$PACKAGE_NAME has been installed. Check the install.log file for more information."
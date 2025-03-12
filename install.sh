#!/bin/bash

# Get ninetails directory path
NINETAILS_DIR=$(dirname $(realpath $0))
printf "NINETAILS_DIR: $NINETAILS_DIR\n"

# Set the package name
PACKAGE_NAME="ninetails"

# Remove the build directory if it exists
if [ -d "$NINETAILS_DIR/build" ]; then
    echo "Removing build directory..."
    rm -rf $NINETAILS_DIR/build
    rm -rf $NINETAILS_DIR/ninetails.egg-info
fi

# Reinstall the package
echo "Installing $PACKAGE_NAME..."
pip install $NINETAILS_DIR

echo "$PACKAGE_NAME has been installed. Check the install.log file for more information."
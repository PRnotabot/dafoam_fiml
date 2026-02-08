#!/bin/bash

# Clean up case directory
rm -rf 0 postProcessing processor* *.bin *.info
rm -rf {1..9}* *.obj log.* *.log *.dat
rm -rf mphys.html OptView.hst opt_*.txt
rm -rf probePointCoords.json
rm -rf sr_training/results sr_training/*.pkl

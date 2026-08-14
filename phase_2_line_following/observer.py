# -----------------------------------------------------------------------------
# QBot Line-Following Observer
# -----------------------------------------------------------------------------
# Launch this on the local Windows machine before starting the robot-side script.

from pal.utilities.probe import Observer

observer = Observer()
observer.add_scope(numSignals=4,
                   name='Motor Speed Plot',
                   signalNames=['Left Motor Cmd', 'Right Motor Cmd', 'Left Motor Meas', 'Right Motor Meas'])
observer.add_scope(numSignals=3,
                   name='Body Speed Plot',
                   signalNames=['Forward Speed', 'Turn Speed', 'Gyro Measurement'])
observer.add_display(imageSize=[200, 320, 1], scalingFactor=2, name='Downward Facing Raw')
observer.launch()

# -----------------------------------------------------------------------------
# QBot Odometry Observer
# -----------------------------------------------------------------------------
# Run this on the local Windows machine before the robot-side script.
# It receives the telemetry streams emitted by the QBot platform.

from pal.utilities.probe import Observer

observer = Observer()
observer.add_scope(numSignals=4,
                   name='Motor Speed Plot',
                   signalNames=['Left Motor Cmd', 'Right Motor Cmd', 'Left Motor Meas', 'Right Motor Meas'])
observer.add_scope(numSignals=3,
                   name='Body Speed Plot',
                   signalNames=['Forward Speed', 'Turn Speed', 'Gyro Measurement'])
observer.launch()

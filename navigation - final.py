import cv2
import depthai as dai
import numpy as np

# for debugging purposes
#import sys
#sys.stdout = open('debug_log.txt', 'w')
#sys.stderr = sys.stdout

pipeline = dai.Pipeline()

color = (255,255,255)

# Create cameras and stereo depth node
monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
stereo = pipeline.create(dai.node.StereoDepth)
spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_ACCURACY)
stereo.setExtendedDisparity(True) # for stuff close the the camera
# stereo.setLeftRightCheck(True)
# stereo.setRectification(True)

# Link mono outputs to stereo
monoLeftOut = monoLeft.requestOutput((640, 400))
monoRightOut = monoRight.requestOutput((640, 400))
monoLeftOut.link(stereo.left)
monoRightOut.link(stereo.right)

stereo.setRectification(True)
stereo.setExtendedDisparity(True)

colorMap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
colorMap[0] = [0, 0, 0]  # to make zero-disparity pixels black (e.g., no depth information or background)

#making a 3*3 grid
num_rows = 3
num_cols = 3

for row in range(num_rows):
    for col in range(num_cols):
        # Normalized coordinates (0.0 to 1.0)
        x_start = col / num_cols
        y_start = row / num_rows
        x_end = (col + 1) / num_cols
        y_end = (row + 1) / num_rows
        
        topLeft = dai.Point2f(x_start, y_start)
        bottomRight = dai.Point2f(x_end, y_end)
        
        config = dai.SpatialLocationCalculatorConfigData()
        config.depthThresholds.lowerThreshold = 10
        config.depthThresholds.upperThreshold = 9000
        config.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
        config.roi = dai.Rect(topLeft, bottomRight)
        spatialLocationCalculator.initialConfig.addROI(config)

spatialLocationCalculator.inputConfig.setWaitForMessage(False)

# Create image filters node and configure it
filter = pipeline.create(dai.node.ImageFilters)
medianFilter = dai.filters.params.MedianFilter.KERNEL_5x5
filter.initialConfig.insertFilter(medianFilter)
stereo.disparity.link(filter.input)
xoutSpatialQueue = spatialLocationCalculator.out.createOutputQueue()
outputDepthQueue = spatialLocationCalculator.passthroughDepth.createOutputQueue()

stereo.depth.link(spatialLocationCalculator.inputDepth)
#stereo.depth.link(filter.input)
#filter.output.link(spatialLocationCalculator.inputDepth) - skipping this one because
# 1. filtered frame + mean give unstable behavior
# 2. median algorithm already handles noise well

inputConfigQueue = spatialLocationCalculator.inputConfig.createInputQueue()

with pipeline:
    pipeline.start()
    while pipeline.isRunning():
        spatialData = xoutSpatialQueue.get().getSpatialLocations()
        outputDepthIMage : dai.ImgFrame = outputDepthQueue.get()

        frameDepth = outputDepthIMage.getCvFrame()
        frameDepth = outputDepthIMage.getFrame()

        depthFrameColor = cv2.normalize(frameDepth, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
        for depthData in spatialData:
            roi = depthData.config.roi
            roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
            xmin = int(roi.topLeft().x)
            ymin = int(roi.topLeft().y)
            xmax = int(roi.bottomRight().x)
            ymax = int(roi.bottomRight().y)

            depthMin = depthData.depthMin
            depthMax = depthData.depthMax

            z = depthData.spatialCoordinates.z
            fontType = cv2.FONT_HERSHEY_TRIPLEX
            if z > 0 and z < 1000:  # obstacle detected within 1m
                print(f"OBSTACLE at Z={int(z)}mm")
                cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), (0,0,255), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
                cv2.putText(depthFrameColor, f"OBSTACLE AT: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymin + 20), fontType, 0.5, (0,0,255))
            else:

                cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
                cv2.putText(depthFrameColor, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymin + 20), fontType, 0.5, color)
                cv2.putText(depthFrameColor, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymin + 35), fontType, 0.5, color)
                cv2.putText(depthFrameColor, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymin + 50), fontType, 0.5, color)

        cv2.imshow("depth", depthFrameColor)

        key = cv2.waitKey(1)
        if key == ord('q'):
            pipeline.stop()
            break
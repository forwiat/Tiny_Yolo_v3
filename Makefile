#WORK=/home/kurata/Projects/VLP64_101_G2E_V2M/work/
WORK=/data2/hoaphan/onnxruntime/
#SDK_ONNX=/opt/poky/2.4.3/sysroots/aarch64-poky-linux/usr/
SDK_ONNX=/data2/hoaphan/onnxruntime/sdk/sysroots/

onnxruntime: tiny_yolo_v3.cpp
    ${CXX} -std=c++14 tiny_yolo_v3.cpp box.cpp image.cpp \
    -DONNX_ML \
    -I ${WORK}tiny_yolo3/src/ \
    -I ${WORK}build/tmp/sysroots-components/aarch64/opencv/usr/include/ \
    -L ${WORK}build/tmp/sysroots-components/aarch64/tbb/usr/lib64/ \
    -L ${WORK}build/tmp/sysroots-components/aarch64/tiff/usr/lib64/ \
    -L ${SDK_ONNX}aarch64-poky-linux/usr/lib64/ \
    -L ${SDK_ONNX}aarch64-poky-linux/usr/lib64/onnx/ \
    -L ${SDK_ONNX}aarch64-poky-linux/usr/lib64/external/protobuf/cmake/ \
    -L ${SDK_ONNX}aarch64-poky-linux/usr/lib64/external/re2/ \
    -L ${WORK}build/tmp/sysroots-components/aarch64/opencv/usr/lib64/ \
    -lonnxruntime_session \
    -lonnxruntime_providers \
    -lautoml_featurizers \
    -lonnxruntime_framework \
    -lonnxruntime_optimizer \
    -lonnxruntime_graph \
    -lonnxruntime_common \
    -lonnx_proto \
    -lprotobuf \
    -lre2 \
    -lonnxruntime_util \
    -lonnxruntime_mlas \
    -lonnx \
    -ljpeg -ltbb -ltiff -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_photo -lopencv_imgcodecs \
    -lpthread -O2 -fopenmp -ldl ${LDFLAGS} -o tiny_yolo_v3

clean:
    rm -rf *.o tiny_yolo_v3

#define WIN32_LEAN_AND_MEAN     // avoids winsock header conflicts
#define NOMINMAX                // prevents Windows headers from defining min and max macros

// Windows communication headers for transferring data over the network
#include <winsock2.h>
#include <ws2tcpip.h>
#include <unknwn.h>
#include <type_traits>

#include <Kinect.h>

// OpenCV headers for image processing and display, core.hpp for basic structures,
// calib3d.hpp for camera calibration, highgui.hpp for GUI functions,
// imgproc.hpp for image processing, aruco_detector.hpp for ArUco marker detection
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

// Libraries for types, algorithms, and data structures
#include <algorithm>
#include <vector>
#include <string>
#include <cstdint>

// Libraries for IO operations, file handling, and string manipulation
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

namespace ArucoConfiguration {
    constexpr int ArucoMarkerID = 0;
    constexpr double ArucoMarkerSize = 100;
    constexpr cv::aruco::PredefinedDictionaryType Dictionary = cv::aruco::DICT_4X4_50;
}

struct Pose6D {
    double translationX;
    double translationY;
    double translationZ;
    double rotationX;
    double rotationY;
    double rotationZ;
};

// Kinect SDK objects are Component Object Models (COMs) and need to be released
// properly to avoid memory leaks. IUNKNOWN is the base interface for all COM objects, and it ensures release() is defined
template<typename T>
void safeRelease(T*& resource) {
    if (resource) {
        static_assert(std::is_base_of<IUnknown, T>::value, "T must be derived from IUnknown");
        resource->Release();
        resource = nullptr;
    }
}

struct CameraCalibration {
// OpenCV offers the Mat type for storing camera matrices that contain intrinsic parameters
// and distortion coefficients.
// https://docs.opencv.org/5.0/main_modules/classcv_1_1Mat.html#mat
    cv::Mat cameraMatrix;
    cv::Mat distortionCoefficients;
// The Camera intrinsic parameters matrix is a 3x3 upper triangular matrix containing
// focal lengths and the principal point offset
// The distortion coefficients matrix contains the radial and tangential distortion parameters.
// These are numerical values associated with the distortion created by
// the camera's lens and imaging sensor
};

CameraCalibration loadCameraCalibration(const std::string& filename) {
    CameraCalibration calibration;

// cv::FileStorage reads and writes XML, YAML, or JSON files containing camera calibration data.
    cv::FileStorage file(filename, cv::FileStorage::READ);
    if (file.isOpened()) {
        file["cameraMatrix"] >> calibration.cameraMatrix;
        file["distortionCoefficients"] >> calibration.distortionCoefficients;

        if (calibration.cameraMatrix.empty() || calibration.distortionCoefficients.empty()) {
            throw std::runtime_error("Calibration data is missing or invalid in file: " + filename);
        }

// OpenCV stores 3dVectors as 64-bit FP numbers, so we convert the coefficients to double for consistency
        calibration.cameraMatrix.convertTo(calibration.cameraMatrix, CV_64F);
        calibration.distortionCoefficients.convertTo(calibration.distortionCoefficients, CV_64F);

        file.release();
    } else {
        throw std::runtime_error("Failed to open camera calibration file: " + filename);
    }

    return calibration;
}

/* 
 Rotation matrix to Euler angles conversion
 Takes a 3x3 rotation matrix defined using the Rodrigues rotation formula as input
 and returns the vector [ RotationX, RotationY, RotationZ ]
 where the entries represent the rotation angles in degrees

 TouchDesigner expects Euler angles in degrees, rather than rotation matrices in radians
*/
cv::Vec3d rotationMatrixToEulerAngles(const cv::Matx33d& RotationMatrix) {
    const double sy = std::sqrt(
        RotationMatrix(0,0) * RotationMatrix(0,0)
        + RotationMatrix(1,0) * RotationMatrix(1,0));
// sy is the scale factor used to check for singularities in the rotation matrix
// a singularity is a situation where the rotation matrix loses a degree of freedom,
// making it impossible to uniquely determine all Euler angles
// if sy is close to zero, it indicates a singularity (gimbal lock) situation
// this means that two of the rotation angles become dependent on each other
// rotation is driven by 3 nested rings, if the second ring rotates close to 90 degrees
// the first and third rings will describe the same rotation, causing the gimbal lock

    double xRadians, yRadians, zRadians = 0.0;

    if (sy > 1e-6) {
// To calculate the euler angles, we need to use the arctan2 functions
// that calculate the angle of a point (y, x) relative to the origin, taking into account the correct quadrant
// https://raw.org/math/linear-algebra/extracting-translation-rotation-and-scaling-from-homogeneous-transformations/
        xRadians = std::atan2(RotationMatrix(2,1), RotationMatrix(2,2));
        yRadians = std::atan2(-RotationMatrix(2,0), sy);
        zRadians = std::atan2(RotationMatrix(1,0), RotationMatrix(0,0));
    }
// if sy is close to zero, namely less than 0.000001, it indicates a singularity, so we hardcode the handling of this case
    else {
// Gimbal lock: 
// Suppose one of the rotations is close to 90 degrees, causing gimbal lock
// Set RotationZ to zero, then calculate RotationX and RotationY from the remaining elements
        xRadians = std::atan2(-RotationMatrix(1,2), RotationMatrix(1,1));
        yRadians = std::atan2(-RotationMatrix(2,0), sy);
        zRadians = 0.0; // no change in RotationZ due to gimbal lock
    }

    return cv::Vec3d(xRadians, yRadians, zRadians) * 180.0 / CV_PI;
}

class UdpSender{
private:
    SOCKET socket_ = INVALID_SOCKET;
    sockaddr_in destination_{};
    bool winsock_initialized = false;

public:
    UdpSender() = default;
    ~UdpSender() {
        close();
    }

    bool open(const std::string& address, int port){
// WSADATA is a structure that contains information about the Windows Sockets implementation
// and it's required for initializing Winsock
// The WSAStartup function initializes the Winsock library using the specified version, namely 2.2
// and fills the WSADATA structure with details about the implementation.
        WSADATA wsaData{};
        if (WSAStartup(MAKEWORD(2,2), &wsaData) != 0) {
            std::cerr << "Failed to initialize Winsock." << std::endl;
            return false;
        }

        winsock_initialized = true;

// Create a UDP socket for sending data
//using the IPv4 address family (AF_INET), datagram type (SOCK_DGRAM), and UDP protocol (IPPROTO_UDP)
        socket_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (socket_ == INVALID_SOCKET) {
            std::cerr << "Failed to create socket with error " << WSAGetLastError() << std::endl;
            close();
            return false;
        }

// Initialize the destination address structure for the UDP socket, including the address family, port, and IP address
        destination_ = {};
        destination_.sin_family = AF_INET;
        destination_.sin_port = htons(port);

// To check if the IP address can be successfully converted to binary form, we use inet_pton,
// which takes the address family, the IP address string, and a pointer to the destination binary address structure,
// and returns a result indicating success or failure.
        const int conversionResult = inet_pton(AF_INET, address.c_str(), &destination_.sin_addr);
        if (conversionResult <= 0) {
            std::cerr << "Failed to convert IP address" << address << std::endl;
            close();
            return false;
        }

        std::cout << "Successfully connected to " << address << ":" << port << std::endl;
        return true;
    }

    bool sendPose(const Pose6D& pose) {
        std::ostringstream stream;
        stream << std::fixed << std::setprecision(3) << pose.translationX << " " << pose.translationY << " " << pose.translationZ << " "
               << pose.rotationX << " " << pose.rotationY << " " << pose.rotationZ;
        
        const std::string message = stream.str();

// Send the pose data as a UDP message to the destination address
// The sendto function takes the socket, message, message length, flags, destination address, and address length
// It returns the number of bytes sent or SOCKET_ERROR on failure
// reinterpret_cast<const sockaddr*>(&destination_) is necessary to cast the destination address to the required sockaddr type
        const int sendResult = sendto(socket_, message.c_str(), static_cast<int>(message.size()), 0,
                                      reinterpret_cast<const sockaddr*>(&destination_), sizeof(destination_));
        if (sendResult == SOCKET_ERROR) {
            std::cerr << "Failed to send pose with error " << WSAGetLastError() << std::endl;
            return false;
        }
        return true;
    }

    void close() {
        if (socket_ != INVALID_SOCKET) {
            closesocket(socket_);
            socket_ = INVALID_SOCKET;
        }

        if (winsock_initialized) {
            WSACleanup();
            winsock_initialized = false;
        }
    }
};

class KinectColorReader{
private:
// https://learn.microsoft.com/en-us/previous-versions/windows/kinect/dn773008(v=ieb.10)
    IKinectSensor* kinectSensor = nullptr;
// https://learn.microsoft.com/en-us/previous-versions/windows/kinect/dn772963(v=ieb.10)
    IColorFrameReader* kinectFrameReader = nullptr;
    int colorWidth = 0;
    int colorHeight = 0;
    std::vector<BYTE> bgraBuffer;

public:
    ~KinectColorReader() {
        close();
    }

    KinectColorReader() = default;

    bool open() {
// HRESULT is a data type used in Windows programming to represent error codes and success/failure status
        HRESULT result = GetDefaultKinectSensor(&kinectSensor);

        if (FAILED(result) || !kinectSensor) {
            std::cerr << "Failed to get default Kinect sensor" << std::endl;
            return false;
        }

        result = kinectSensor->Open();

        if (FAILED(result)) {
            std::cerr << "Failed to open Kinect sensor" << std::endl;
            return false;
        }
        
// https://learn.microsoft.com/en-us/previous-versions/windows/kinect/dn772967(v=ieb.10)
        IColorFrameSource* colorSource = nullptr;

        result = kinectSensor->get_ColorFrameSource(&colorSource);
        if (FAILED(result) || !colorSource) {
            std::cerr << "Failed to get color frame source" << std::endl;
            
            safeRelease(colorSource);
            close();
            return false;
        }
        
// https://learn.microsoft.com/en-us/previous-versions/windows/kinect/dn772989(v=ieb.10)
        IFrameDescription* frameDescription = nullptr;
        
        result = colorSource->get_FrameDescription(&frameDescription);
        if (FAILED(result) || !frameDescription) {
            std::cerr << "Failed to get frame description" << std::endl;
            
            safeRelease(frameDescription);
            safeRelease(colorSource);
            close();
            return false;
        }

        frameDescription->get_Width(&colorWidth);
        frameDescription->get_Height(&colorHeight);

// The buffer size is calculated based on the color frame width, height, and 4 bytes per pixel
        bgraBuffer.resize(static_cast<std::size_t>(colorWidth) *
                        static_cast<std::size_t>(colorHeight) * 4);

        result = colorSource->OpenReader(&kinectFrameReader);

        safeRelease(frameDescription);
        safeRelease(colorSource);

        if (FAILED(result) || !kinectFrameReader)
        {
            std:: cerr << "Failed to open Kinect color frame reader." << std::endl;
            
            close();
            return false;
        }

        std::cout << "Kinect color stream: " << colorWidth << " x " << colorHeight << std::endl;
        return true;
    }

// Read takes the reference to an OpenCV matrix object and fills it with the latest color frame data from the Kinect sensor.
    bool read(cv::Mat& outputBGR) {
        if (!kinectFrameReader) {
            return false;
        }

        IColorFrame* colorFrame = nullptr;
        HRESULT result = kinectFrameReader->AcquireLatestFrame(&colorFrame);
        if (FAILED(result) || !colorFrame) {
            safeRelease(colorFrame);
            return false;
        }

// CopyConvertedFrameDataToArray copies the color frame data to the specified array
// using the size of the array cast as UINT, the array, and the color format
        result = colorFrame->CopyConvertedFrameDataToArray(static_cast<UINT>(bgraBuffer.size()),
                                             bgraBuffer.data(), ColorImageFormat_Bgra);
        safeRelease(colorFrame);
        if (FAILED(result)) {
            return false;
        }

// This Mat constructor is defined as (row, columns, data type, data, optional step size)
// and DOES NOT copy the data, it just wraps the existing buffer into a Mat object
// CV_8UC4 represents an 8-bit unsigned integer matrix with 4 channels (BGRA)
// cvtColor converts the image from one color space to another, in this case from BGRA to BGR
// https://docs.opencv.org/5.0/main_modules/imgproc_color_conversions.html#cvtcolor
        cv::Mat bgraImage(colorHeight, colorWidth, CV_8UC4, bgraBuffer.data());
        cv::cvtColor(bgraImage, outputBGR, cv::COLOR_BGRA2BGR);

        return true;
    }

    void close() {
        safeRelease(kinectFrameReader);
        if (kinectSensor) {
            kinectSensor->Close();
        }
        safeRelease(kinectSensor);
    }
};

// https://docs.opencv.org/4.13.0/d5/dae/tutorial_aruco_detection.html
class ArucoPoseTracker {
private:
    cv::Mat cameraMatrix;
    cv::Mat distortionCoefficients;
// https://docs.opencv.org/4.13.0/d5/d0b/classcv_1_1aruco_1_1Dictionary.html
    cv::aruco::Dictionary arucoDictionary;
// https://docs.opencv.org/4.13.0/d1/dcd/structcv_1_1aruco_1_1DetectorParameters.html
    cv::aruco::DetectorParameters arucoParameters;
// https://docs.opencv.org/4.13.0/d2/d1a/classcv_1_1aruco_1_1ArucoDetector.html
    cv::aruco::ArucoDetector arucoDetector;
    std::vector<cv::Point3d> markerObjectPoints;

public:
    ArucoPoseTracker(const cv::Mat& cameraMatrix, const cv::Mat& distortionCoefficients) :
        cameraMatrix(cameraMatrix.clone()),
        distortionCoefficients(distortionCoefficients.clone()),
        arucoDictionary(cv::aruco::getPredefinedDictionary(ArucoConfiguration::Dictionary)),
        arucoParameters(),
        arucoDetector(arucoDictionary, arucoParameters)
    {
// Using the half size to place the origin at the center of the marker
// The quadrants are organized as Second (top-right), First (top-left), Third (bottom-left), Fourth (bottom-right) in the XY plane
        const double markerHalfSize = ArucoConfiguration::ArucoMarkerSize / 2.0;
        markerObjectPoints = {
            cv::Point3d(-markerHalfSize,  markerHalfSize, 0),
            cv::Point3d( markerHalfSize,  markerHalfSize, 0),
            cv::Point3d( markerHalfSize, -markerHalfSize, 0),
            cv::Point3d(-markerHalfSize, -markerHalfSize, 0)
        };
    }

// Estimates the pose of the Aruco marker in the input image and outputs the 6D pose
// rejectedCandidates are the detected marker corners that are not the real markers
// which is useful for debugging and refining detection parameters
    bool estimatePose(cv::Mat& inputImage, Pose6D& outputPose) {
        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners;

        std::vector<std::vector<cv::Point2f>> rejectedCandidates;

// Aruco's Detector.detectMarkers searches the image for square candidates, decodes their binary patterns,
// sees if they match the dictionary, and returns their coordinates, IDs, and an optional vector of discarded candidates
// https://docs.opencv.org/4.13.0/d2/d1a/classcv_1_1aruco_1_1ArucoDetector.html#a0c1d14251bf1cbb06277f49cfe1c9b61
        arucoDetector.detectMarkers(inputImage, markerCorners, markerIds, rejectedCandidates);

        if (markerIds.empty()) {
            return false;
        }

// Draw detected markers for debbuging purposes
// https://docs.opencv.org/4.13.0/de/d67/group__objdetect__aruco.html#ga2ad34b0f277edebb6a132d3069ed2909
        cv::aruco::drawDetectedMarkers(inputImage, markerCorners, markerIds);

        const auto targetIterator = std::find(markerIds.begin(), markerIds.end(), ArucoConfiguration::ArucoMarkerID);
        if (targetIterator == markerIds.end()) {
            return false;
        }

        const std::size_t targetIndex = static_cast<std::size_t>(std::distance(markerIds.begin(), targetIterator));

        cv::Vec3d rotationVector, translationVector;

// The Perspective-n-Point is the problem that arises when trying to determine the pose of a calibrated camera
// given a set of 3D points and their corresponding 2D projections in the image.
// OpenCV provides the solvePnP function to solve the Perspective-n-Point problem given the 3D points from the Aruco marker
// and the camera Calibration parameters
        const bool poseSolved = cv::solvePnP(
            markerObjectPoints,
            markerCorners[targetIndex],
            cameraMatrix,
            distortionCoefficients,
            rotationVector,
            translationVector,
            false,
            cv::SOLVEPNP_IPPE_SQUARE
        );

        if (!poseSolved) {
            return false;
        }

// drawFramesAxes for debugging visualization
// https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html#gab3ab7bb2bdfe7d5d9745bb92d13f9564
        cv::drawFrameAxes(
            inputImage,
            cameraMatrix,
            distortionCoefficients,
            rotationVector,
            translationVector,
            ArucoConfiguration::ArucoMarkerSize * 0.75,
            2
        );

        cv::Mat rotationMatrix;
        cv::Rodrigues(rotationVector, rotationMatrix);

// Convert the rotation vector to the explicit 3x3 rotation matrix that will be used to extract the Euler angles
        const cv::Matx33d explicitRotationMatrix(
            rotationMatrix.at<double>(0, 0), rotationMatrix.at<double>(0, 1), rotationMatrix.at<double>(0, 2),
            rotationMatrix.at<double>(1, 0), rotationMatrix.at<double>(1, 1), rotationMatrix.at<double>(1, 2),
            rotationMatrix.at<double>(2, 0), rotationMatrix.at<double>(2, 1), rotationMatrix.at<double>(2, 2)
        );

        const cv::Vec3d eulerAngles = rotationMatrixToEulerAngles(explicitRotationMatrix);

        outputPose.translationX = translationVector[0];
        outputPose.translationY = translationVector[1];
        outputPose.translationZ = translationVector[2];
        outputPose.rotationX = eulerAngles[0];
        outputPose.rotationY = eulerAngles[1];
        outputPose.rotationZ = eulerAngles[2];

        return true;
    }
};

int main(int argc, char* argv[]) {
    try {

// Let's suppose we have a calibration file named "calibration.yml" in the working directory
        const std::string calibrationFilename = "calibration.yml";
        const CameraCalibration calibration = loadCameraCalibration(calibrationFilename);


// Initialize the Kinect color reader and the UDP sender
        KinectColorReader kinect;

        if (!kinect.open()) {
            std::cerr << "Failed to open Kinect." << std::endl;
            return 1;
        }

        UdpSender udp;
        
// Replace this hardcoded IP and port with the actual IP and port of the UDP server
        if (!udp.open("127.0.0.1", 9000)) {
            std::cerr << "Failed to open UDP connection." << std::endl;
            return 1;
        }

// Initialize the Aruco pose tracker with the camera calibration parameters
// cv::namedWindow is a function provided by OpenCV to create a window for displaying images on the screen
        ArucoPoseTracker tracker(calibration.cameraMatrix, calibration.distortionCoefficients);

        cv::namedWindow("Kinect Color Reader", cv::WINDOW_NORMAL);

// Main loop for reading frames from the Kinect and processing them
// Loop breaks when the ESC key is pressed
// On failure to read a frame, the loop awaits for key input and continues if the ESC key is not pressed
        while (true) {
            cv::Mat frame;

            const bool frameReadSuccessfully = kinect.read(frame);

            if (!frameReadSuccessfully) {
                const int key = cv::waitKey(1);
                if (key == 27) { // ESC key = ASCII 27
                    break;
                }
                continue;
            }

// Initialize the pose structure to store the estimated pose of the Aruco marker
// Then estimate the pose using the Aruco pose tracker, and if successful, send it via UDP and display it on the frame
// cv::putText is used to display the pose information on the frame, essentialy for debugging and visualization purposes
            Pose6D pose;

            const bool tracking = tracker.estimatePose(frame, pose);

            if (tracking) {
                udp.sendPose(pose);

                std::ostringstream poseText;

                poseText << std::fixed << std::setprecision(2)
                         << "Translation (mm): [" << pose.translationX << ", " << pose.translationY << ", " << pose.translationZ << "] "
                         << "Rotation (deg): [" << pose.rotationX << ", " << pose.rotationY << ", " << pose.rotationZ << "]";
                cv::putText(frame, poseText.str(), cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            }

            else {
                cv::putText(frame, "Aruco marker not detected", cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            }

            cv::imshow("Kinect Color Reader", frame);
            const int key = cv::waitKey(1);
            if (key == 27) { // Stated before: ESC key = ASCII 27
                break;
            }

        }
        cv::destroyAllWindows();
        kinect.close();
        udp.close();
        return 0;
        }
// Catch any exceptions thrown during the execution of the program and go on with your life
    catch (const std::exception& exception) {
        std::cerr << "Exception: " << exception.what() << std::endl;
        return 1;
    }
}
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/ccalib.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/ximgproc/lsc.hpp>

#include "feature_extraction.h"

using namespace std;
using namespace cv;

int threshold1 = 549;//349;//466;//506;
int threshold2 = 269;//193;//258;//229;

void cluster(Mat &image, Mat &output, int K, vector<int> &labels, Mat1f &centers) {
    TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 10, 1.);
    
    Mat toCluster;
    Mat gaussed = image.clone();
    GaussianBlur(image, gaussed, Size(5, 5), 2);
    
    gaussed.convertTo(toCluster, CV_32F);
    toCluster = toCluster.reshape(1, toCluster.total());
    
    kmeans(toCluster, K, labels, criteria, 1, KMEANS_PP_CENTERS, centers);
    cout << centers << endl;
    
    centers = centers.reshape(gaussed.channels(), centers.rows);
    toCluster = toCluster.reshape(gaussed.channels(), toCluster.rows);
    
    Vec3f *p = toCluster.ptr<Vec3f>();
    for (size_t i = 0; i < toCluster.rows; i++) {
        int center_index = labels[i];
        p[i] = centers.at<Vec3f>(center_index);
    }
    
    output = toCluster.reshape(gaussed.channels(), gaussed.rows);
    output.convertTo(output, CV_8U);
    
    
    //medianBlur(output, output, 23);
    /*Mat kernel = Mat(3, 3, CV_8U);
    erode(output, output, kernel);
    dilate(output, output, kernel);*/
    
}

//genero i colori
vector<Vec3b> creaColori(Mat img, Mat1f centers, int K)
{
    vector<Vec3b> colori;
    /*for (int i = 0; i < K; i++)
    {
        //recupero la posizione del centro
        int x = centers[i][0];
        int y = centers[i][1];
        cout << "centro: " << x << " " << y << endl;

        //recupero colore dall'immagine originale in quella posizione
        cout << "colore: " << img.at<Vec3b>(x, y) << endl;
        colori.push_back(img.at<Vec3b>(x, y));
    }*/



    //creo colori random
    for (int i = 0; i < K; i++)
    {
        int b = theRNG().uniform(0, 255);
        int g = theRNG().uniform(0, 255);
        int r = theRNG().uniform(0, 255);

        colori.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }
    return colori;
}

//k means con posizione e colore in hsv
vector<Mat> kMeansHsv(Mat img, int K) {
    vector<int> labels;
    Mat1f centers;

    vector<Point3f> pixels;
    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);

    //creo vettore di punti con posizione e valore di colore preso da hsv
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            Point3f actualPixel(i, j, hsv.at<Vec3b>(i, j)[0]); //prendo il value del colore
            pixels.push_back(actualPixel);
        }
    }
    cout << "pixel: " << pixels[164] << endl;

    vector<int> newLabels;
    kmeans(pixels, K, newLabels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.), 10, KMEANS_PP_CENTERS, centers);
    cout << "centers: " << centers << endl;

    //prendo colori dall'immagine originale
    vector<Vec3b> colori;
    colori = creaColori(img, centers, K);

    //scorro tutta l'immagine e coloro diversamente a seconda della label
    int count = 0;
    Mat clusterImg = img.clone();
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            clusterImg.at<Vec3b>(i, j) = colori[newLabels[count]];
            count++;
        }
    }

    imshow("immagine", clusterImg);


    //voglio mettere ogni cluster su uno sfondo bianco
    vector<Mat> clusters;
    for (int x = 0; x < K; x++)
    {
        Mat sfondo(Size(img.cols, img.rows), img.type());
        count = 0;
        for (int i = 0; i < sfondo.rows; i++)
        {
            for (int j = 0; j < sfondo.cols; j++)
            {
                if (newLabels[count] == x)
                {
                    sfondo.at<Vec3b>(i, j) = img.at < Vec3b>(i, j);
                }
                else
                {
                    sfondo.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
                }
                count++;
            }
        }

        clusters.push_back(sfondo);
    }
    
    return clusters;
}

void cluster2(Mat &input, Mat &output, int K, vector<int> &labels, Mat1f &centers) {
    TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 100, 0.01);
    Mat image = input.clone();
    //GaussianBlur(input, image, Size(7, 7), 2);
    float rateo = image.cols / image.rows;
    float posMul = 1;
    float positionMulX = posMul;
    float positionMulY = posMul ;
    vector<Vec6f> to_cluster;
    Mat hsv;
    cvtColor(image, hsv, COLOR_BGR2HSV);
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            float x = static_cast<float>(j) / posMul;/// image.cols * positionMulX;
            float y = static_cast<float>(i) / posMul;/// image.rows * positionMulY;
            
            Vec3b color = image.at<Vec3b>(i, j);
            Vec6f point = Vec6f(x, y, color[0], color[1], color[2], -1);
            
            to_cluster.push_back(point);
        }
    }
    
    kmeans(to_cluster, K, labels, criteria, 1, KMEANS_PP_CENTERS, centers);
    cout << centers << endl;
    
    output = image.clone();
    
    vector<Vec3b> colors;
    
    for (int i = 0; i < K; i++) {
        int x = static_cast<float>(centers[i][0]) * posMul;//* image.cols / positionMulX;
        int y = static_cast<float>(centers[i][1]) * posMul;//* image.rows / positionMulY;
        
        float b = centers[i][2];
        float g = centers[i][3];
        float r = centers[i][4];
        
        Vec3b color = Vec3b(b, g, r);
        colors.push_back(color);
        
        cout << "x: " << x << " y: " << y << " color: " << color << endl;
    }
    
    
    size_t arrayIndex = 0;
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            int center_id = labels[arrayIndex];
            Vec3f center = colors[center_id];
            output.at<Vec3b>(i, j) = center;
            arrayIndex++;
        }
    }
    
}

void clusterFeatures(Mat &input, Mat &output, int K, vector<int> &labels, Mat1f &centers, vector<KeyPoint> &keypoints) {
    Mat image = input.clone();
    output = input.clone();
    //GaussianBlur(input, image, Size(7, 7), 2);
    //medianBlur(image, image, 7);
    
    vector<Point3f> points;
    cv::Ptr<cv::FeatureDetector> featureExtractor = cv::xfeatures2d::SiftFeatureDetector::create();
    
    //vector<KeyPoint> keypoints;
    featureExtractor->detect(image, keypoints);
    
    Mat hsv;
    cvtColor(image, hsv, COLOR_BGR2HSV);
    for (int i = 0; i < keypoints.size(); i++) {
        Point2f pt = keypoints[i].pt;
        int x = static_cast<int>(pt.x);
        int y = static_cast<int>(pt.y);
        //Vec3b color = hsv.at<Vec3b>(x, y);
        //points.push_back(Point3f(pt.x, pt.y, color[0]));
        float response = keypoints[i].size;
        points.push_back(Point3f(pt.x, pt.y, response));
        
        //Vec3b color = image.at<Vec3b>(pt.x, pt.y);
        //Vec6f point = Vec6f(pt.x, pt.y, color[0], color[1], color[2], -1);
        //points.push_back(point);
        
    }
    
    TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 10, 1.);
    kmeans(points, K, labels, criteria, 1, KMEANS_PP_CENTERS, centers);
    cout << centers << endl;
    
    
    vector<Vec3b> clusterColors;
    for (int i = 0; i < K; i++) {
        float b = theRNG().uniform(0, 255);
        float g = theRNG().uniform(0, 255);
        float r = theRNG().uniform(0, 255);
        clusterColors.push_back(Vec3b(b, g, r));
    }
    
    for (int i = 0; i < points.size(); i++) {
        int center_id = labels[i];
        Vec3b color = clusterColors[center_id];
        //circle(output, points[i], 5, color, -1);
        circle(output, Point2f(points[i].x, points[i].y), 5, color, -1);
        //circle(output, Point2f(points[i][0], points[i][1]), 5, color, -1);
    }
    
    Mat drawKeypointsImage;
    drawKeypoints(image, keypoints, drawKeypointsImage);
    imshow("Draw Keypoints", drawKeypointsImage);
}

void drawRect(Mat &image, vector<KeyPoint> keypoints, Scalar color = Scalar(0, 255, 0), uint rect_width = 5) {
    float minX = FLT_MAX;
    float maxX = FLT_MIN;
    float minY = FLT_MAX;
    float maxY = FLT_MIN;
    Point2f pointMinX, pointMaxX, pointMinY, pointMaxY;
    for (int i = 0; i < keypoints.size(); i++) {
        Point2f point = keypoints[i].pt;
        if (point.x < minX) {
            pointMinX = point;
            minX = point.x;
        }
        if (point.y < minY) {
            pointMinY = point;
            minY = point.y;
        }
        if (point.x > maxX) {
            pointMaxX = point;
            maxX = point.x;
        }
        if (point.y > maxY) {
            pointMaxY = point;
            maxY = point.y;
        }
    }
    rectangle(image, Point2f(minX, minY), Point2f(maxX, maxY), color, rect_width);
}

void drawRect2(Mat image, Mat cluster, Scalar color = Scalar(0, 255, 0), uint rect_width = 5) {
    float minX = FLT_MAX;
    float maxX = FLT_MIN;
    float minY = FLT_MAX;
    float maxY = FLT_MIN;
    Point2f pointMinX, pointMaxX, pointMinY, pointMaxY;
    for (int i = 0; i < cluster.rows; i++) {
        for (int j = 0; j < cluster.cols; j++) {
            Vec3b color = cluster.at<Vec3b>(i, j);
            if (color == Vec3b(255, 255, 255)) continue;
            
            if (j < minX) {
                minX = j;
            }
            if (j > maxX) {
                maxX = j;
            }
            if (i < minY) {
                minY = i;
            }
            if (i > maxY) {
                maxY = i;
            }
        }
    }
    
    rectangle(image, Point2f(minX, minY), Point2f(maxX, maxY), color, rect_width);
}

void applyDerivative(Mat input, Mat &output) {
   
    Mat image = input.clone();
    Mat gaussed, gaussed2;
    GaussianBlur(image, gaussed, Size(7, 7), 1.3);
    GaussianBlur(image, gaussed2, Size(7, 7), 2);
    
    
    image.convertTo(image, CV_32F);
    gaussed.convertTo(gaussed, CV_32F);
    gaussed2.convertTo(gaussed2, CV_32F);
    Mat res = image - gaussed;
    res.convertTo(output, CV_8UC3);
    //dilate(output, output, Mat());
    cvtColor(output, output, COLOR_BGR2GRAY);
    threshold(output, output, 0, 255, THRESH_BINARY | THRESH_OTSU);
    
    
    
    medianBlur(output, output, 3);
    
}

void splitCluster(Mat image, Mat clustered, Mat1f centers, vector<Mat> &clusters) {
    cv::Ptr<cv::FeatureDetector> featureExtractor = cv::xfeatures2d::SiftFeatureDetector::create();
    for (int k = 0; k < centers.rows; k++) {
        Vec3b clusterColor;
        if (centers.cols == 6) {
            Vec6f ctr = centers.at<Vec6f>(k);
            clusterColor = Vec3b(ctr[2], ctr[3], ctr[4]);
        } else {
            clusterColor = Vec3b(centers.at<Vec3f>(k));
        }
        
        cout << "cluster color: " << clusterColor << endl;
        Mat mat = Mat(image.rows, image.cols, image.type(), Scalar(255, 255, 255));
        for (int i = 0; i < image.rows; i++) {
            for (int j = 0; j < image.cols; j++) {
                Vec3b color = clustered.at<Vec3b>(i, j);
                //cout << "Color: " << color << endl;
                if (color == clusterColor) {
                    mat.at<Vec3b>(i, j) = image.at<Vec3b>(i, j);
                }
            }
        }
        clusters.push_back(mat);
        
        vector<KeyPoint> keypoints;
        featureExtractor->detect(mat, keypoints);
        //drawKeypoints(mat, keypoints, mat);
        
        //imshow("Clustered " + to_string(k), mat);
    }
    
}

int main() {
    Mat mat;
    vector<String> images_paths;
    utils::fs::glob("../dataset", "*.*", images_paths);
    vector<Mat> images;
    for (String imPath : images_paths) {
        Mat mat = imread(imPath);
        //GaussianBlur(mat, mat, Size(7, 7), 2);
        if (mat.empty()) {
            continue;
        }
        images.push_back(mat);
    }
    
    //the dictionary should be trained with 11 images to get better results
    int vocabularyImagesSize = 30;
    
    vector<Mat> vocabularyImages;
    for (int i = 0; i < vocabularyImagesSize; i++) {
        int randomIndex = theRNG().uniform(0, images.size());
        vocabularyImages.push_back(images[randomIndex]);
        
    }
    
    int trainingImagesSize = images.size();
    vector<Mat> trainingSetImages;
    for (int i = 0; i < trainingImagesSize; i++) {
        trainingSetImages.push_back(images[i]);
        //imshow("Img" + to_string(i), images[i]);
    }
    
    Mat testImage = imread("../benchmark/Figure 26.jpg");
    
    TreeFeatureExtractor treeFeature = TreeFeatureExtractor(200, "vocabulary_test", "model_test");
    treeFeature.createVocabulary(vocabularyImages);
    treeFeature.train(trainingSetImages);
    int res = treeFeature.predict(testImage);
    if (res == 1) {
        cout << "There's a tree" << endl;
    } else {
        cout << "No trees in this image" << endl;
    }
    
    Mat clstrdimg;
    Mat derivateImage;
    applyDerivative(testImage, derivateImage);
    //imshow("image", testImage);
    imshow("derivative", derivateImage);
    vector<int> lbls;
    Mat1f cntrs;
    Mat toCluster;
    uint numberColors = 8;
    cluster2(testImage, clstrdimg, numberColors, lbls, cntrs);
    imshow("Clustered image", clstrdimg);
    vector<Mat> clusters;
    splitCluster(testImage, clstrdimg, cntrs, clusters);
    for (int i = 0; i < clusters.size(); i++) {
        Mat cluster = clusters[i];
        int res = treeFeature.predict(cluster);
        if (res == 1) {
            drawRect2(testImage, cluster);
            //cout << "cluster " + to_string(i) << " is a tree" << endl;
            imshow("Cluster with tree " + to_string(i), cluster);
        }
    }
    
    /*Mat clusteredImage;
    uint numclusters = 2;
    vector<int> clusterLabels;
    Mat1f centers;
    vector<KeyPoint> clusterKeypoints;
    clusterFeatures(testImage, clusteredImage, numclusters, clusterLabels, centers, clusterKeypoints);
    imshow("clustered", clusteredImage);
    
    vector<vector<KeyPoint>> clustered_keypoints;
    for (int i = 0; i < numclusters; i++) {
        clustered_keypoints.push_back(vector<KeyPoint>());
    }
    for (int i = 0; i < clusterKeypoints.size(); i++) {
        int center_id = clusterLabels[i];
        
        clustered_keypoints[center_id].push_back(clusterKeypoints[i]);
    }
    
    bool containsTree[numclusters];
    for (int i = 0; i < numclusters; i++) {
        Mat descriptors;
        
        
        int predictedLabel = treeFeature.predict(testImage, clustered_keypoints[i]);
        containsTree[i] = predictedLabel == 1;
    }
    
    vector<Point2f> treePoints;
    for(int i = 0; i < numclusters; i++) {
        if (containsTree[i]) {
            for (int j = 0; j < clustered_keypoints[i].size(); j++) {
                treePoints.push_back(clustered_keypoints[i][j].pt);
                drawRect(testImage, clustered_keypoints[i]);
            }
        }
    }*/
    
    /*for (int i = 0; i < treePoints.size(); i++) {
        circle(testImage, treePoints[i], 5, Scalar(255, 0, 0), -1);
    }
    
    for (int i = 0; i < boundPoints.size(); i++) {
        circle(testImage, boundPoints[i], 5, Scalar(0, 0, 255), -1);
    }*/
    imshow("test", testImage);
    
    waitKey(0);
    return 0;
}


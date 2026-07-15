//
// Created by HP on 2026/7/4.
//
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>

#include "gdlz/trt/cv/ocr/trt_ocr_process.h"

using gdlz::trt::icv::ocr::TensorRTCvOCRProcesserReC;
using gdlz::trt::icv::ocr::TensorRTCvOCRProcesserDet;


int TensorRTCvOCRProcesserDet::MatToProImg(cv::Mat& img, ProImg& pro_img){
    pro_img.org_mat  = std::make_unique<cv::Mat>(img);
    pro_img.blob_mat = std::make_unique<cv::Mat>();
    return 0;
}

int TensorRTCvOCRProcesserDet::PreProcess(ProImg& img, PreParams params){
    if (!img.org_mat || !img.blob_mat){
        return -1;
    }
    cv::Mat res;
    cv::resize(*img.org_mat,res,
        cv::Size(params.img_width,
                      params.img_height)
    );
    cv::dnn::blobFromImage(
        res,
        *img.blob_mat, params.scalefactor,
        cv::Size(),
        cv::Scalar(), false, false
    );
    return 0;
}

int TensorRTCvOCRProcesserDet::PostProcess(float* output,ProImg& img, PostParams params, RectBoxs& rect_boxs) {
    if (!rect_boxs.rect_mat){
        rect_boxs.rect_mat=std::make_unique<std::vector<cv::Rect>>();
    }
    rect_boxs.rect_mat->clear();
    cv::Mat probMap(params.img_height, params.img_width, CV_32F);
    cv::Mat(params.img_height, params.img_width, CV_32F, output).copyTo(probMap);

    double p_min, p_max;
    cv::minMaxLoc(probMap, &p_min, &p_max);
    if (p_max > p_min) {
        probMap = (probMap - p_min) / (p_max - p_min);
    }
    cv::Mat bin;
    cv::threshold(probMap, bin, params.thresh, 1.0, cv::THRESH_BINARY);
    bin.convertTo(bin, CV_8U, 255);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(bin, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    const double min_area = std::max(5.0, 30.0 *
        img.org_mat->cols * img.org_mat->rows / (params.img_width * params.img_height)
    );

    std::vector<cv::RotatedRect> textBoxes;
    for (const auto& c : contours) {
        double area = cv::contourArea(c);
        if (area < min_area) continue;
        cv::RotatedRect r = cv::minAreaRect(c);
        // unclip 膨胀：放大宽高，保持中心和角度不变
        r.size.width  *= params.unclip_ratio;
        r.size.height *= params.unclip_ratio;
        textBoxes.push_back(r);
    }

    float sx = static_cast<float>(img.org_mat->cols) / static_cast<float>(params.img_width);
    float sy = static_cast<float>(img.org_mat->rows) / static_cast<float>(params.img_height);

    for (const auto& rect : textBoxes) {
        // 1. 映射到原图坐标
        cv::Point2f center(rect.center.x * sx, rect.center.y * sy);
        cv::Size2f size(rect.size.width * sx, rect.size.height * sy);
        cv::RotatedRect scaled_rect(center, size, rect.angle);

        // 2. 取轴对齐包围盒 + 留白，直接从原图裁剪
        cv::Rect bbox = scaled_rect.boundingRect();
        bbox.x      = std::max(0, bbox.x - params.PAD);
        bbox.y      = std::max(0, bbox.y - params.PAD);
        bbox.width  = std::min(img.org_mat->cols - bbox.x, bbox.width + 2 * params.PAD);
        bbox.height = std::min(img.org_mat->rows - bbox.y, bbox.height + 2 * params.PAD);
        if (bbox.width <= 0 || bbox.height <= 0) continue;
        cv::rectangle(*img.org_mat, bbox, cv::Scalar(0, 0, 255), 2);
        rect_boxs.rect_mat->push_back(bbox);
    }
    return 0;
}

int TensorRTCvOCRProcesserReC::MatToProImg(cv::Mat& img, ProImg& pro_img){
    pro_img.org_mat  = std::make_unique<cv::Mat>(img);
    pro_img.blob_mat = std::make_unique<cv::Mat>();
    return 0;
}

int TensorRTCvOCRProcesserReC::PreProcess(ProImg& img,PreParams params){
    if (!img.org_mat || !img.blob_mat){
        return -1;
    }
    cv::Mat res;
    cv::resize(*img.org_mat,res,
        cv::Size(params.img_width,
                      params.img_height)
    );
    cv::dnn::blobFromImage(
        res,
        *img.blob_mat, params.scalefactor,
        cv::Size(),
        cv::Scalar(), false, false
    );
    return 0;
}

int TensorRTCvOCRProcesserReC::PostProcess(float* output,ProImg& img, PostParams params, RectBoxs& rect_boxs) {

    return 0;
}

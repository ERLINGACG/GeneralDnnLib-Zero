//
// Created by HP on 2025/12/9.
//
#include "gdlz/cv/framework/yolo_framework.h"
#include "dylog/logger.h"

using gdlz::cv_framework::YoloFramework;
using gdlz::resources::YoloResourceDirector;
using dylog::DynamicLogger;

YoloFramework& YoloFramework::ResourceDirector(YoloResourceDirector& director)
{

    director.SetConfPath(this->conf_path).Hand(this->resource);
    DynamicLogger().
        setInvokeName("YoloFramework")
        .debug("conf_path:{}",director.conf_path)
        .debug(
                "model_path:{}",this->resource.model_path.c_str()
        );
    // dylog::logger().info("model_path:{}",this->resource.model_path);
    return *this;
}

YoloFramework& YoloFramework::Input(cv::Mat& input, param::YoloParam& param)
{
    param.input=std::make_unique<cv::Mat>(input);
    return *this;
}

YoloFramework& YoloFramework::Input(int size, unsigned char* byte, param::YoloParam& param)
{
    const std::vector buf(byte,byte+size);
    param.input=std::make_unique<cv::Mat>(cv::imdecode(buf, cv::IMREAD_COLOR));
    return *this;
}

YoloFramework& YoloFramework::SetBlob(param::YoloParam& param)
{
    param.blob=std::make_unique<cv::Mat>(
       cv::dnn::blobFromImage(
           *param.input,
           1.0 / 255.0,
           cv::Size(this->resource.input_width, this->resource.input_height),
           cv::Scalar(0, 0, 0),
           true,
           false
       )
    );
    this->resource.Net->setInput(*param.blob);
    return *this;
}

YoloFramework& YoloFramework::Forward(param::YoloParam& param)
{
    param.forward_output=std::make_unique<cv::Mat>(this->resource.Net->forward());
    return *this;
}

YoloFramework& YoloFramework::Process(param::YoloParam& param)
{
    int numChannels = param.forward_output->size[1];  // 84（4坐标 + 80类别）
    int numAnchors = param.forward_output->size[2];   // 8400（候选框数量）
    auto* data = param.forward_output->ptr<float>(0);
    float x_factor = static_cast<float>(param.input->cols) / static_cast<float>(this->resource.input_width);
    float y_factor = static_cast<float>(param.input->rows) / static_cast<float>(this->resource.input_height);
    for (int w = 0; w < numAnchors; w++)
    {

        float maxScore = 0.0f;
        int classId = -1;
        // 查找当前框的最大类别置信度（索引 4~83 对应 80 个类别）
        for (int h = 4; h < numChannels; h++)
        {
            float score = data[h * numAnchors + w]; // 索引 4~83 对应类别置信度
            // cout<<"score = "<<score<<" classId = "<<h-4<<endl;
            if (score > maxScore)
            {
                maxScore = score;
                classId = h - 4;  // 类别 ID（0~79）
            }
            if (maxScore > 0.5)
            {
                float cx = data[0 * numAnchors + w];  // 中心 x
                float cy = data[1 * numAnchors + w];  // 中心 y
                float w_box = data[2 * numAnchors + w];// 框宽
                float h_box = data[3 * numAnchors + w];// 框高
                int x1 = static_cast<int>((cx - w_box / 2) * x_factor);
                int y1 = static_cast<int>((cy - h_box / 2) * y_factor);
                int x2 = static_cast<int>((cx + w_box / 2) * x_factor);
                int y2 = static_cast<int>((cy + h_box / 2) * y_factor);

                param.boxes->emplace_back(cv::Rect(x1, y1, x2 - x1, y2 - y1));
                param.class_ids->emplace_back(classId);
                param.confidences->emplace_back(maxScore);
            }
        }
    }

    return *this;
}

YoloFramework& YoloFramework::PostProcess(param::YoloParam& param)
{
    cv::dnn::NMSBoxes(
            *param.boxes,
            *param.confidences,
            this->resource.conf_thresh,
            this->resource.nms_thresh,
            *param.indices
    );

    auto describe=std::map<std::string,int>();
    for (const int index : *param.indices) {
        const auto class_id = (*param.class_ids)[index];

        const auto box = (*param.boxes)[index];
        const auto confidences = (*param.confidences)[index];
        cv::rectangle(
            *param.input,
            box,
            cv::Scalar(0, 255, 0),  // 颜色不变（绿色）
            1,                  // 线条粗细从2→1（细线条更锐利，无模糊感）
            cv::LINE_AA         // 抗锯齿（核心！消除框的锯齿模糊）
        );
        std::string text = this->resource.class_names->at(class_id) + ":" + std::to_string(confidences);
        describe[this->resource.class_names->at(class_id)]++;
        // 2. 文字位置（原位置，避免超出图像）
        cv::Point text_org(box.x, box.y - 5);
        // 3. 文字参数（和putText保持一致）
        constexpr int font_face = cv::FONT_HERSHEY_SIMPLEX;
        constexpr double font_scale = 0.5;
        constexpr int thickness = 1;
        int baseline = 0;

        // 4. 获取文字尺寸（宽、高）
        cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);// 5. 计算背景矩形的范围（避免文字超出图像边界）
        cv::Rect bg_rect;
        bg_rect.x = text_org.x;                          // 矩形左上角x
        bg_rect.y = text_org.y - text_size.height - 5;   // 矩形左上角y（上移5像素避免和框重叠）
        bg_rect.width = text_size.width;                 // 矩形宽度（文字宽度）
        bg_rect.height = text_size.height + baseline + 5;// 矩形高度（文字高度+基线+上下边距）

        // 6. 修正矩形边界（防止超出图像）
        bg_rect.x = std::max(0, bg_rect.x);
        bg_rect.y = std::max(0, bg_rect.y);
        bg_rect.width = std::min(bg_rect.width, param.input->cols - bg_rect.x);
        bg_rect.height = std::min(bg_rect.height, param.input->rows - bg_rect.y);

        // 7. 绘制蓝色背景（填充矩形，Scalar(B, G, R)，蓝色是(255,0,0)）
        cv::rectangle(*param.input, bg_rect, cv::Scalar(255, 0, 0), cv::FILLED);
        // 8. 绘制文字（建议用白色，和蓝色背景对比明显）
        cv::putText(
            *param.input,
            text,
            text_org,
            font_face,
            font_scale,
            cv::Scalar(255, 255, 255),  // 文字颜色：白色（替换原绿色，避免和蓝色融合）
            thickness,
            cv::LINE_AA
        );
    }
    auto describe_str=nlohmann::json(describe).dump();
    param.describe_str=std::make_unique<std::string>(describe_str);
    return *this;
}

YoloFramework& YoloFramework::GetOutput(data::YoloOutput& output, const param::YoloParam& param)
{

    output.SetData(*param.input,*param.describe_str);
    return *this;
}



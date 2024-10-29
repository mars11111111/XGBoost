import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import xgboost as xgb

# 加载模型
model = joblib.load('best_model.pkl')

# 获取模型输入特征数量及顺序
model_input_features = [ 'A2',  'A3', 'A5', 'A6', 'B3','B4','B5','smokeG', ' exerciseG1', ' exerciseG2', ' exerciseG3',  '年龄',  '工龄','上岗时间', '工时分组','生活满意度', '抑郁症状级别', '睡眠状况','疲劳蓄积程度']
expected_feature_count = len(model_input_features)

# 定义新的特征选项及名称
cp_options = {
    0: '无症状 (0)',
    1: '轻度职业紧张 (2)',
    2: '中度职业紧张 (3)',
    3: '重度职业紧张 (4)'
}

# Streamlit界面设置
st.title("职业紧张预测")

# 年龄输入
age = st.number_input("年龄：", min_value=1, max_value=120, value=50)


# 工龄输入
service_years = st.number_input("工龄：", min_value=1, max_value=120, value=50)

# 近一个月平均每天加班时间输入，对应B3
overtime_hours = st.number_input("近一个月平均每天加班时间：", min_value=1, max_value=120, value=50）

# A2（性别）选择
A2_options = {1: '男性', 2: '女性'}
A2 = st.selectbox(
    "性别：",
    options=list(A2_options.keys()),
    format_func=lambda x: A2_options[x]
)

# A3（学历）选择
A3_options = {1: '初中及以下', 2: '高中或中专', 3: '大专或高职', 4: '大学本科', 5: '研究生及以上'}
A3 = st.selectbox(
    "学历：",
    options=list(A3_options.keys()),
    format_func=lambda x: A3_options[x]
)

# A5（月收入）选择
A5_options = {1: '少于3000元', 2: '3000 - 4999元', 3: '5000 - 6999元', 4: '7000 - 8999元', 5: '9000 - 10999元', 6: '11000元及以上'}
A5 = st.selectbox(
    "月收入：",
    options=list(A5_options.keys()),
    format_func=lambda x: A5_options[x]
)

# A6默认值设为0
A6_default = 0

# B4（是否轮班）选择
B4_options = {1: '否', 2: '是'}
B4 = st.selectbox(
    "是否轮班：",
    options=list(B4_options.keys()),
    format_func=lambda x: B4_options[x]
)

# B5（是否需要上夜班）选择
B5_options = {1: '否', 2: '是'}
B5 = st.selectbox(
    "是否需要上夜班：",
    options=list(B5_options.keys()),
    format_func=lambda x: B5_options[x]
)

# smoke（是否吸烟）选择
smoke_options = {1: '是的', 2: '以前吸，但现在不吸了', 3: '从不吸烟'}
smoke = st.selectbox(
    "是否吸烟：",
    options=list(smoke_options.keys()),
    format_func=lambda x: smoke_options[x]
)

# 工时分组选择
working_hours_group_options = {1: '35到40小时', 2: '40到48小时', 3: '48到54小时', 4: '54到105小时'}
working_hours_group = st.selectbox(
    "工时分组：",
    options=list(working_hours_group_options.keys()),
    format_func=lambda x: working_hours_group_options[x]
)

# exerciseG1默认值设为0
exerciseG1_default = 0

# exerciseG1默认值设为0
exerciseG1_default = 0

# exercise（是否有进行持续至少30分钟的中等强度锻炼）选择
exercise_options = {1: '无', 2: '偶尔，1 - 3次/月', 3: '有，1~3次/周', 4: '经常，4~6次/周', 5: '每天'}
exercise = st.selectbox(
    "是否有进行持续至少30分钟的中等强度锻炼：",
    options=list(exercise_options.keys()),
    format_func=lambda x: exercise_options[x]
)

# 生活满意度滑块
life_satisfaction = st.slider("生活满意度（1 - 5）：", min_value=1, max_value=5, value=3)

# 睡眠状况滑块
sleep_status = st.slider("睡眠状况（1 - 5）：", min_value=1, max_value=5, value=3)

# 疲劳积蓄程度滑块
work_load = st.slider("疲劳积蓄程度（1 - 5）：", min_value=1, max_value=5, value=3)

# 抑郁症状级别滑块
depression_level = st.slider("抑郁症状级别（1 - 5）：", min_value=1, max_value=5, value=3)

# 定义默认特征值为0的特征列表
missing_features = ['A6', 'exerciseG1', 'exerciseG2'，'onboarding_time']
default_feature_values = {feature: 0 for feature in missing_features}

def predict():
    """
    进行职业紧张预测并生成建议和可视化。
    """
    try:
        # 获取用户输入
        user_inputs = {
            '年龄': age,
            'A2': A2,
            'A3': A3,
            'A5': A5,
            'A6': A6_default,
            'B3': overtime_hours,
            'B4': B4,
            'B5': B5,
            'smokeG': smoke,
            'exerciseG1': exerciseG1_default, 
            'exerciseG2': exerciseG2_default, 
            'exerciseG3': exercise_options[exercise],
            '工龄':  service_years,
            '上岗时间': 0,
            '工时分组': working_hours_group,
            '生活满意度': life_satisfaction,
            '抑郁症状级别': depression_level,
            '睡眠状况': sleep_status,
            '疲劳蓄积程度': work_load
        }

        # 按照固定顺序整理特征值
        feature_values = [user_inputs[feature] for feature in model_input_features]
        features_array = np.array([feature_values])

        if len(features_array[0])!= expected_feature_count:
            # 如果特征数量不匹配，使用零填充来达到模型期望的特征数量
            padded_features = np.pad(features_array, ((0, 0), (0, expected_feature_count - len(features_array[0]))), 'constant')
            predicted_class = model.predict(padded_features)[0]
            predicted_proba = model.predict_proba(padded_features)[0]
        else:
            predicted_class = model.predict(features_array)[0]
            predicted_proba = model.predict_proba(features_array)[0]

        # 将数字对应转换为文本及格式化概率输出
        category_mapping = {'无职业紧张症状': 0, '轻度职业紧张症状': 1, '中度职业紧张症状': 2, '重度职业紧张症状': 3}
        predicted_category = [k for k, v in category_mapping.items() if v == predicted_class][0]
        probability_labels = ['无职业紧张症状', '轻度职业紧张症状', '中度职业紧张症状', '重度职业紧张症状']
        formatted_probabilities = [f'{prob:.4f}' for prob in predicted_proba]
        probability_output = [f"{label}: '{probability}'" for label, probability in zip(probability_labels, formatted_probabilities)]

        # 显示预测结果
        st.write(f"**预测类别：** {predicted_category}")
        st.write(f"**预测概率：** {dict(zip(probability_labels, formatted_probabilities))}")

        # 根据预测结果生成建议
        probability = predicted_proba[predicted_class] * 100
        print(f"原始概率值：{predicted_proba[predicted_class]}")
        print(f"计算后的概率值：{probability}")

        if predicted_category == '无职业紧张症状':
            advice = (
                f"根据我们的模型，该员工无职业紧张症状。"
                f"模型预测该员工无职业紧张症状的概率为 {probability:.2f}%。"
                "请继续保持良好的工作和生活状态。"
            )
        elif predicted_category == '轻度职业紧张症状':
            advice = (
                f"根据我们的模型，该员工有轻度职业紧张症状。"
                f"模型预测该员工职业紧张程度为轻度的概率为 {probability:.2f}%。"
                "建议您适当调整工作节奏，关注自身身心健康。"
            )
        elif predicted_category == '中度职业紧张症状':
            advice = (
                f"根据我们的模型，该员工有中度职业紧张症状。"
                f"模型预测该员工职业紧张程度为中度的概率为 {probability:.2f}%。"
                "建议您寻求专业帮助，如心理咨询或与上级沟通调整工作。"
            )
        elif predicted_category == '重度职业紧张症状':
            advice = (
                f"根据我们的模型，该员工有重度职业紧张症状。"
                f"模型预测该员工职业紧张程度为重度的概率为 {probability:.2f}%。"
                "强烈建议您立即采取行动，如休假、寻求医疗支持或与管理层协商改善工作环境。"
            )
        else:
            advice = "预测结果出现未知情况。"

        st.write(advice)

        # 进行SHAP值计算，不直接使用DMatrix
        if len(features_array[0])!= expected_feature_count:
            data_df = pd.DataFrame(padded_features[0].reshape(1, -1), columns=model_input_features)
        else:
            data_df = pd.DataFrame(features_array[0].reshape(1, -1), columns=model_input_features)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data_df)

        # 更加谨慎地处理expected_value
        base_value = explainer.expected_value if not isinstance(explainer.expected_value, list) else (
            explainer.expected_value[0] if len(explainer.expected_value) > 0 else None)
        if base_value is None:
            raise ValueError("Unable to determine base value for SHAP force plot.")

        try:
            shap.plots.force(base_value, shap_values[0], data_df)
            plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
        except Exception as e:
            print(f"Error in force plot: {e}")
            # 如果force plot失败，尝试其他绘图方法
            fonts = fm.findSystemFonts(fontpaths=None, fontext='ttf')
            font_names = [fm.FontProperties(fname=fname).get_name() for fname in fonts]
            if 'SimHei' in font_names:
                plt.rcParams['font.sans-serif'] = ['SimHei']
            elif 'Microsoft YaHei' in font_names:
                plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
            else:
                plt.rcParams['font.sans-serif'] = [font_names[0]] if font_names else ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            shap.summary_plot(shap_values, data_df, show=False)
            plt.title('SHAP值汇总图')
            plt.xlabel('特征')
            plt.ylabel('SHAP值')
            plt.savefig("shap_summary_plot.png", bbox_inches='tight', dpi=1200)

        st.image("shap_summary_plot.png")
    except Exception as e:
        st.write(f"出现错误：{e}")


# 添加预测按钮
if st.button("预测"):
    predict()

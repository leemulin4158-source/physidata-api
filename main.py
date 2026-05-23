from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from scipy import stats
import google.generativeai as genai
import math
import os

app = FastAPI(title="PhysiData Lab API")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# 你的 API Key (已為你保留)
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

def get_available_model():
    print("正在尋找可用的 AI 模型...")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ 找到可用模型: {m.name}")
            return m.name
    return 'gemini-pro'

selected_model_name = get_available_model()
model = genai.GenerativeModel(selected_model_name)
print(f"🚀 已成功連接並使用模型: {selected_model_name}")

class DataPayload(BaseModel):
    x_data: list[float]
    y_data: list[float]
    x_min_scale: float
    y_min_scale: float
    mode: str = "repeated"

class AnalysisPayload(BaseModel):
    slope: float
    intercept: float
    r_squared: float

@app.post("/api/calculate")
async def calculate_data(payload: DataPayload):
    try:
        x = np.array(payload.x_data)
        y = np.array(payload.y_data)
        n = len(x)
        if n < 2: raise HTTPException(status_code=400, detail="至少需要兩筆數據")

        # --- 判斷模式，設定參數 ---
        is_repeated = (payload.mode == "repeated")
        ddof_val = 1 if is_repeated else 0
        den_str = f"{n}-1" if is_repeated else f"{n}"
        
        std_name = "樣本標準差" if is_repeated else "母體標準差"
        std_sym_y = "s_y" if is_repeated else "\\sigma_y"
        word_sym_y = "s_y" if is_repeated else "σ_y"
        
        reg_title_math = "趨勢線模型" if is_repeated else "線性回歸模型"
        reg_title_text = "趨勢線" if is_repeated else "線性回歸"

        # 基礎數學運算
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        sum_sq_x = np.sum((x - mean_x)**2)
        sum_sq_y = np.sum((y - mean_y)**2)
        
        var_x = np.var(x, ddof=ddof_val)
        var_y = np.var(y, ddof=ddof_val)
        s_x = np.std(x, ddof=ddof_val)
        s_y = np.std(y, ddof=ddof_val)

        # 線性回歸與殘差運算
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        y_pred = slope * x + intercept
        sse = np.sum((y - y_pred)**2) # 殘差平方和 (Sum of Squared Errors)

        sign = "+" if intercept >= 0 else "-"
        abs_k = abs(intercept)
        equation_math = f"y = {slope:.4f}x {sign} {abs_k:.4f}"

        # ==========================================
        # 模組化組裝輸出區塊
        # ==========================================
        if is_repeated:
            # 【重複數據模式】：專注於趨勢線與測量不確定度
            uA_y = s_y / math.sqrt(n)
            uB_y = payload.y_min_scale / (2 * math.sqrt(3))
            uc_y = math.sqrt(uA_y**2 + uB_y**2)

            reg_math_block = f"""&\\textbf{{【 {reg_title_math} 】}} \\\\
&\\text{{實驗方程式: }} {equation_math} \\\\
&\\text{{斜率 }} m = {slope:.4f} \\pm {std_err:.4f} \\\\
&\\text{{截距 }} k = {intercept:.4f}"""
            
            y_math_block = f"""\\\\ \\\\
&\\textbf{{【 Y軸 數據分析過程 】}} \\\\
&\\text{{a. 平均值: }} \\bar{{y}} = \\frac{{\\sum y_i}}{{{n}}} = {mean_y:.4f} \\\\
&\\text{{b. 離均差平方和: }} \\sum (y_i - \\bar{{y}})^2 = {sum_sq_y:.4f} \\\\
&\\text{{c. {std_name}: }} {std_sym_y} = \\sqrt{{\\frac{{\\sum (y_i - \\bar{{y}})^2}}{{{den_str}}}}} = \\sqrt{{\\frac{{{sum_sq_y:.4f}}}{{{den_str}}}}} = {s_y:.4f} \\\\
&\\text{{d. A類不確定度: }} u_{{A,y}} = \\frac{{{std_sym_y}}}{{\\sqrt{{n}}}} = \\frac{{{s_y:.4f}}}{{\\sqrt{{{n}}}}} = {uA_y:.4f} \\\\
&\\text{{e. B類不確定度: }} u_{{B,y}} = \\frac{{\\Delta y}}{{2\\sqrt{{3}}}} = \\frac{{{payload.y_min_scale}}}{{2\\sqrt{{3}}}} = {uB_y:.4f} \\\\
&\\text{{f. 組合不確定度: }} u_{{c,y}} = \\sqrt{{u_{{A,y}}^2 + u_{{B,y}}^2}} = \\sqrt{{{uA_y:.4f}^2 + {uB_y:.4f}^2}} = {uc_y:.4f}"""

            # Text blocks for Repeated Mode
            reg_latex_block = f"""% --- {reg_title_text} ---
% 實驗方程式: {equation_math}
m = {slope:.4f} \\pm {std_err:.4f}
k = {intercept:.4f}"""
            reg_word_block = f"""[{reg_title_text}]
實驗方程式: {equation_math}
m = {slope:.4f} ± {std_err:.4f}
k = {intercept:.4f}"""
            y_latex_block = f"""\n\n% --- Y軸 標準差與不確定度 ({std_name}) ---
\\bar{{y}} = {mean_y:.4f}
\\sum (y_i - \\bar{{y}})^2 = {sum_sq_y:.4f}
{std_sym_y} = \\sqrt{{\\frac{{\\sum (y_i - \\bar{{y}})^2}}{{{den_str}}}}} = {s_y:.4f}
u_{{A,y}} = \\frac{{{std_sym_y}}}{{\\sqrt{{n}}}} = \\frac{{{s_y:.4f}}}{{\\sqrt{{{n}}}}} = {uA_y:.4f}
u_{{B,y}} = \\frac{{\\Delta y}}{{2\\sqrt{{3}}}} = \\frac{{{payload.y_min_scale}}}{{2\\sqrt{{3}}}} = {uB_y:.4f}
u_{{c,y}} = \\sqrt{{u_{{A,y}}^2 + u_{{B,y}}^2}} = \\sqrt{{{uA_y:.4f}^2 + {uB_y:.4f}^2}} = {uc_y:.4f}"""
            y_word_block = f"""\n\n[Y軸 標準差與不確定度 ({std_name})]
y的平均值 = {mean_y:.4f}
y的離均差平方和 = {sum_sq_y:.4f}
{std_name} {word_sym_y} = √[ ∑(y_i - y_avg)^2 / ({den_str}) ] = {s_y:.4f}
u_(A,y) = {word_sym_y} / √n = {s_y:.4f} / √{n} = {uA_y:.4f}
u_(B,y) = Δy / (2√3) = {payload.y_min_scale} / (2√3) = {uB_y:.4f}
u_(c,y) = √(u_(A,y)^2 + u_(B,y)^2) = √({uA_y:.4f}^2 + {uB_y:.4f}^2) = {uc_y:.4f}"""

        else:
            # 【不同變因模式】：保留 R^2、顯示變異數、標準差，並新增【殘差平方和 (SSE)】
            reg_math_block = f"""&\\textbf{{【 {reg_title_math} 】}} \\\\
&\\text{{實驗方程式: }} {equation_math} \\\\
&\\text{{斜率 }} m = {slope:.4f} \\pm {std_err:.4f} \\\\
&\\text{{截距 }} k = {intercept:.4f} \\\\
&\\text{{相關係數 }} r = {r_value:.4f} \\\\
&\\text{{決定係數 }} R^2 = {r_value**2:.4f}"""
            
            y_math_block = f"""\\\\ \\\\
&\\textbf{{【 數據離散程度與殘差分析 】}} \\\\
&\\text{{a. X變異數: }} \\sigma_x^2 = \\frac{{\\sum (x_i - \\bar{{x}})^2}}{{{den_str}}} = \\frac{{{sum_sq_x:.4f}}}{{{den_str}}} = {var_x:.4f} \\\\
&\\text{{b. X標準差: }} \\sigma_x = \\sqrt{{\\sigma_x^2}} = {s_x:.4f} \\\\
&\\text{{c. Y變異數: }} \\sigma_y^2 = \\frac{{\\sum (y_i - \\bar{{y}})^2}}{{{den_str}}} = \\frac{{{sum_sq_y:.4f}}}{{{den_str}}} = {var_y:.4f} \\\\
&\\text{{d. Y標準差: }} \\sigma_y = \\sqrt{{\\sigma_y^2}} = {s_y:.4f} \\\\
&\\text{{e. 預測值: }} \\hat{{y}}_i = {slope:.4f}x_i {sign} {abs_k:.4f} \\\\
&\\text{{f. 殘差平方和: }} SSE = \\sum (y_i - \\hat{{y}}_i)^2 = {sse:.4f}"""

            # Text blocks for Variable Mode
            reg_latex_block = f"""% --- {reg_title_text} ---
% 實驗方程式: {equation_math}
m = {slope:.4f} \\pm {std_err:.4f}
k = {intercept:.4f}
r = {r_value:.4f}
R^2 = {r_value**2:.4f}"""
            reg_word_block = f"""[{reg_title_text}]
實驗方程式: {equation_math}
m = {slope:.4f} ± {std_err:.4f}
k = {intercept:.4f}
r = {r_value:.4f}
R^2 = {r_value**2:.4f}"""
            y_latex_block = f"""\n\n% --- 數據離散程度與殘差分析 ---
\\sigma_x^2 = \\frac{{\\sum (x_i - \\bar{{x}})^2}}{{{den_str}}} = {var_x:.4f}
\\sigma_x = \\sqrt{{\\sigma_x^2}} = {s_x:.4f}
\\sigma_y^2 = \\frac{{\\sum (y_i - \\bar{{y}})^2}}{{{den_str}}} = {var_y:.4f}
\\sigma_y = \\sqrt{{\\sigma_y^2}} = {s_y:.4f}
SSE = \\sum (y_i - \\hat{{y}}_i)^2 = {sse:.4f}"""
            y_word_block = f"""\n\n[數據離散程度與殘差分析]
X軸變異數 σ_x^2 = {var_x:.4f}
X軸標準差 σ_x = {s_x:.4f}
Y軸變異數 σ_y^2 = {var_y:.4f}
Y軸標準差 σ_y = {s_y:.4f}
殘差平方和 SSE = {sse:.4f}"""

        # 組合最終輸出字串
        rendered_math = f"""$$
\\begin{{aligned}}
{reg_math_block}
{y_math_block}
\\end{{aligned}}
$$"""
        latex_text = f"{reg_latex_block}{y_latex_block}"
        word_text = f"{reg_word_block}{y_word_block}"

        return {
            "status": "success",
            "regression": {
                "slope": round(slope, 4), "intercept": round(intercept, 4),
                "r_squared": round(r_value**2, 4), "slope_err": round(std_err, 4)
            },
            "rendered_math": rendered_math,
            "latex_text": latex_text,
            "word_text": word_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze")
async def analyze_data(payload: AnalysisPayload):
    try:
        prompt = f"""
        你是一位專業且鼓勵學生的物理探究與實作老師。
        學生剛剛完成了一組實驗數據分析，這是一組線性回歸結果：
        - 最佳擬合斜率 (m): {payload.slope}
        - 截距 (c): {payload.intercept}
        - 相關係數 (R²): {payload.r_squared}
        
        請用繁體中文，針對這組數據的「線性程度 (根據 R² 判斷)」與「潛在的系統誤差 (根據截距判斷)」給出一段約 150 字的客觀分析與建議。語氣要專業但親切，不要使用 Markdown 標題，直接輸出純文字段落即可。
        """
        response = model.generate_content(prompt)
        return {"status": "success", "analysis": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
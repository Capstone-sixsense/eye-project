import os
import json
from cleanvision import Imagelab
from PIL import Image

#CleanVision을 이용한 1차적인 이미지 필터 기능입니다
#check_image_quality()함수와 PassNonPass()함수로 구성되며
#check_image_quality()에서 이미지 분석을 작업하고
#PassNonPass()에서 임계값에 따른 통과 여부를 결정합니다


def check_image_quality(data_path: str, target_filename: str):
    # 분석할 타겟 파일의 절대 경로만 구합니다.
    target_abs_path = os.path.abspath(os.path.join(data_path, target_filename))
    
    # 폴더 전체가 아닌, 해당 파일 하나만 리스트로 전달합니다.
    imagelab = Imagelab(filepaths=[target_abs_path])

    imagelab.find_issues()

    all_issues = imagelab.issues

    try:
        # 결과 테이블에서 해당 파일의 행만 추출합니다.
        file_report = all_issues.loc[target_abs_path].to_dict()
        
        return {
            "filename": target_filename,
            "issues_found": file_report,
            "summary": imagelab.issue_summary.to_dict()
        }
    except KeyError:
        return {"error": f"파일 {target_filename}을 찾을 수 없습니다. 이름이나 경로를 확인하세요."}


def PassNonPass(result: dict):
    #result가 없거나 내부에 'error'가 포함되어 있다면 즉시 불합격 처리
    if not result or "error" in result:
        error_msg = result.get("error", "분석할 데이터가 존재하지 않습니다.")
        return {
            "is_acceptable": False, 
            "messages": [error_msg] # 오타 내용을 그대로 메시지에 담아줍니다.
        }

    thresholds = {
        "blurry": 0.64,           # 0.85보다 낮으면 흐리다고 판단 (기존 0.5 대비 대폭 강화)
        "dark": 0.5,             # 조금만 어두워도 감지
        "low_information": 0.64,  # 정보량이 부족한 이미지 차단
        "odd_aspect_ratio": 0.62, # 비율이 조금만 이상해도 차단
        "light" : 0.90             # 강한 조명 반사에 매우 엄격하게 설정
    }

    error_messages = {
        "blurry": "사진이 너무 흐립니다.",
        "dark": "주변이 너무 어둡습니다.",
        "light": "빛 반사가 너무 강합니다.",
        "low_information": "안저 정보가 부족합니다.",
        "odd_aspect_ratio": "이미지 비율이 부적절합니다."
    }

    #정상적인 경우에만 issues를 읽어옵니다.
    issues_data = result.get("issues_found", {})
    fail_messages = []

    for issue, limit in thresholds.items():
        score_key = f"{issue}_score"
        score = issues_data.get(score_key, 1.0) #점수가 없으면 1.0(정상)으로 간주
        
        #점수가 기준치보다 낮으면(품질이 낮으면) 불합격
        if score < limit:
            fail_messages.append(f"{error_messages[issue]} (점수: {score:.2f})")

    is_acceptable = len(fail_messages) == 0
    
    return {
        "is_acceptable": is_acceptable,
        "messages": fail_messages if not is_acceptable else ["사용 가능한 이미지입니다."]
    }


#AI 모델에 적합한 사이즈로 이미지 크기 조절 함수(초기 설정값 : 1024x1024)
def resize_image_high_quality(input_path: str, output_path: str, target_size=(1024, 1024)):
    """
    고품질 리사이징: 가로세로 비율을 유지하며 LANCZOS 필터 적용
    """
    with Image.open(input_path) as img:
        #원본 비율 유지하며 최대 크기에 맞추기 (thumbnail은 원본을 직접 변경하므로 copy 권장)
        img_copy = img.copy()
        
        #비율을 유지하며 target_size 내에 꽉 차도록 조절합니다.
        img_copy.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        #저장 (의료용은 무손실 PNG 또는 고품질 JPEG 권장)
        #JPEG인 경우 quality를 95 이상으로 설정하여 손실 최소화
        if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
            img_copy.save(output_path, "JPEG", quality=95, subsampling=0)
        else:
            img_copy.save(output_path, "PNG")
            
    return output_path


#자체 테스트용 코드
if __name__ == "__main__":
    TEST_FOLDER = "storage"
    TEST_FILE = "ambiguous_01.png"

    print(f"🔍 {TEST_FILE} 분석을 시작합니다...")

    #품질 분석 수행
    raw_data = check_image_quality(TEST_FOLDER, TEST_FILE)

    #raw_data값 체크
    print(raw_data)

    #판정 수행
    final_report = PassNonPass(raw_data)

    #결과 출력
    print("\n" + "="*30)

    #raw_data가 에러가 아닐 때만 filename을 가져오고, 아니면 TEST_FILE 출력
    display_name = raw_data.get('filename', TEST_FILE)
    
    print(f"파일명: {display_name}")
    print(f"합격 여부: {'✅ PASS' if final_report['is_acceptable'] else '❌ FAIL'}")
    print(f"상세 메시지: {final_report['messages']}")
    
    #세부 내용 출력
    print("\n[항목별 세부 점수 (Scores Only)]")
    issues_found = raw_data.get("issues_found", {})
    
    for key, value in issues_found.items():
        #숫자(int, float)이면서 키 이름에 "_score"가 포함된 경우만 출력
        if isinstance(value, (int, float)) and "_score" in key:
            print(f"  > {key:<25} : {value:.4f}")
            
    print("="*30)


from cleanvision import Imagelab

def check_image_quality(data_path: str, target_filename: str):
    #Imagelab 인스턴스 생성 및 분석 실행
    imagelab = Imagelab(data_path=data_path)
    imagelab.find_issues()

    #모든 이미지에 대한 상세 점수가 들어있는 테이블
    all_issues = imagelab.issues
    
    #특정 파일명에 해당하는 데이터만 필터링
    target_path = os.path.join(data_path, target_filename)
    
    try:
        #해당 파일의 행(row)만 추출
        file_report = all_issues.loc[target_path]
        
        #결과 정리 (딕셔너리 형태로 변환)
        result = {
            "filename": target_filename,
            "issues_found": file_report.to_dict(),
            "summary": imagelab.issue_summary.to_dict() # 참고용 전체 요약
        }
        return result
        
    except KeyError:
        return {"error": f"파일 {target_filename}을 찾을 수 없습니다."}

def PassNonPass(result: dict):

    if not result:
        return {"is_acceptable": False, "messages": ["분석할 이미지 데이터를 찾을 수 없습니다."]}

    error_messages = {
        "is_blurry_issue": "사진이 너무 흐립니다.",
        "is_dark_issue": "주변이 너무 어둡습니다.",
        "is_light_issue": "빛이 너무 강합니다.",
        "is_low_information_issue": "이미지에 정보가 부족합니다.",
        "is_grayscale_issue": "컬러 사진이 아닙니다."
    }

    issues = result.get("issues_found", {})
    fail_messages = []

    # True인 이슈들만 골라 메시지 리스트에 담기
    for issue_key, is_detected in issues.items():
        if is_detected is True and issue_key in error_messages:
            fail_messages.append(error_messages[issue_key])

    # 최종 판단: 메시지가 없으면 합격(True)
    is_acceptable = len(fail_messages) == 0
    
    return {
        "is_acceptable": is_acceptable,
        "messages": fail_messages if not is_acceptable else ["사용 가능한 이미지입니다."]
    }



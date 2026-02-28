import re
import ast

def clean_text(text):
    """특수문자 제거 및 공백 정규화"""
    if not text: return ""
    text = re.sub(r"[\[\]\'\"\.]", "", str(text))
    return re.sub(r"\s+", " ", text).strip()

def parse_response(task_type, raw_response, metadata=None):
    if not raw_response or raw_response == "Error":
        return "Error", {"parsing_failed": True, "reason": "empty_or_error_input"}
    
    if task_type == 'character':
        if '</think>' in raw_response:
            raw_response = raw_response.split('</think>')[-1].strip()

        if '</thought>' in raw_response:
            raw_response = raw_response.split('</thought>')[-1].strip()

        if 'assistantfinal' in raw_response:
            raw_response = raw_response.split('assistantfinal')[-1].strip()

        val_a, val_b = "", ""
        
        # 1-2. 줄 단위로 A, B 답변 추출
        for line in raw_response.split('\n'):
            line = line.strip()
            # A: 또는 B: 로 시작하는지 확인
            is_a = re.search(r'^A\s*[:\.]', line, re.IGNORECASE)
            is_b = re.search(r'^B\s*[:\.]', line, re.IGNORECASE)
            
            if is_a or is_b:
                # [수정됨] 대괄호 [...]가 몇 개 있는지 확인
                brackets_found = re.findall(r'\[(.*?)\]', line)
                
                if len(brackets_found) > 1:
                    # 대괄호가 2개 이상이면 즉시 에러 반환
                    return "Error", {
                        "parsing_failed": True, 
                        "reason": "ambiguous_multiple_brackets", 
                        "line_content": line
                    }
                elif len(brackets_found) == 1:
                    # 딱 하나만 있으면 그 값을 추출
                    extracted_val = brackets_found[0].strip()
                else:
                    # 대괄호가 없으면 prefix(A: 등) 제거 후 나머지 전체 사용
                    extracted_val = re.sub(r'^[AB]\s*[:\.]', '', line, flags=re.IGNORECASE).strip()
                
                if is_a: val_a = extracted_val
                if is_b: val_b = extracted_val

        # A, B 둘 중 하나라도 없으면 에러
        if not val_a or not val_b:
            return "Error", {"parsing_failed": True, "reason": "format_mismatch_missing_AB", "raw": raw_response}

        # 1-4. 메타데이터 기반 유효성 검증
        if metadata:
            # (기존 로직과 동일)
            # 메타데이터에서 N1, N2 가져오기 (문자열이 아닌 리스트일 수도 있으니 안전하게 처리 필요하면 수정 가능)
            # 여기서는 metadata가 dict이고 값이 문자열이라고 가정
            entity_n1 = metadata.get('N1_entity', '').strip()
            entity_n2 = metadata.get('N2_entity', '').strip()
            
            # 유효한 후보군 set 생성
            valid_candidates = {entity_n1, entity_n2}
            valid_candidates.discard("") 

            def validate_choice(extracted_text, candidates):
                if not candidates: # 후보군이 없으면 검증 패스 (혹은 에러 처리)
                    return extracted_text, None
                    
                # 추출된 텍스트 안에 후보군(N1, N2)이 포함되어 있는지 확인
                found_matches = [c for c in candidates if c in extracted_text]
                
                # 룰: 딱 하나만 들어있어야 함
                if len(found_matches) == 1:
                    return found_matches[0], None
                
                if len(found_matches) > 1:
                    return None, "ambiguous_multiple_matches_in_text"
                
                if not found_matches:
                    return None, "no_match_found_in_candidates"

                return None, "unknown_error"

            final_a, err_a = validate_choice(val_a, valid_candidates)
            final_b, err_b = validate_choice(val_b, valid_candidates)
            
            if err_a or err_b:
                error_reason = err_a if err_a else err_b
                return "Error", {
                    "parsing_failed": True, 
                    "reason": "validation_mismatch",
                    "detail": error_reason,
                    "expected": list(valid_candidates),
                    "got_raw": f"A:[{val_a}] / B:[{val_b}]"
                }
            
            val_a = final_a
            val_b = final_b

        return f"{val_a}/{val_b}", None
        

    # ==========================================================================
    # 2. Plot Task
    # ==========================================================================
    elif task_type == 'plot':
        cleaned_response = raw_response
        print("Raw response: ", cleaned_response)
        
        # 1. 불필요한 태그/사고 과정 제거
        if '</think>' in cleaned_response:
            cleaned_response = cleaned_response.split('</think>')[-1].strip()

        if '</thought>' in cleaned_response:
            cleaned_response = cleaned_response.split('</thought>')[-1].strip()

        if 'assistantfinal' in cleaned_response:
            cleaned_response = cleaned_response.split('assistantfinal')[-1].strip()
        
        matches = re.findall(r'\b([123])\b', cleaned_response)
        print("matches found: ", matches)
        
        answer_map = metadata.get('answer_map_obj')
        
        if isinstance(answer_map, str):
            try:
                answer_map = ast.literal_eval(answer_map)
            except:
                pass
                
        if answer_map:
            if len(matches) == 1:
                # 숫자가 딱 하나만 발견된 경우 -> 성공
                selected_num = matches[0]
                parsed_type = answer_map.get(selected_num, "OutOfRange")
                print(f"Selected: {selected_num} -> {parsed_type}")
                return parsed_type, {"selected_num": selected_num}
            
            elif len(matches) > 1:
                # 숫자가 여러 개 발견된 경우 -> 모호하므로 에러 처리
                print(f"ParsingError: Multiple numbers found {matches}")
                return "ParsingError", {
                    "parsing_failed": True, 
                    "reason": "Ambiguous output (multiple numbers found)",
                    "found_matches": matches,
                    "cleaned_response": cleaned_response[:200]
                }
            else:
                # 숫자가 발견되지 않은 경우
                return "ParsingError", {
                    "parsing_failed": True, 
                    "reason": "No number found",
                    "cleaned_response": cleaned_response[:200]
                }
        else:
            return "ParsingError", {"parsing_failed": True, "reason": "No answer_map provided"}

    # ==========================================================================
    # 3. KoBBQ Task (A/B/C 추출 → 선택지 인덱스 또는 문자 반환)
    # ==========================================================================
    elif task_type == 'kobbq':
        response = str(raw_response).strip().upper()
        # A, B, C 중 하나만 추출
        match = re.search(r'\b([ABC])\b', response)
        if match:
            return match.group(1), {"parsed_letter": match.group(1)}
        return "Error", {"parsing_failed": True, "reason": "no_valid_ABC", "raw": raw_response}

    return None, None
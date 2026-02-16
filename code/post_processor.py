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
            is_a = re.search(r'^A\s*[:\.]', line, re.IGNORECASE)
            is_b = re.search(r'^B\s*[:\.]', line, re.IGNORECASE)
            
            if is_a or is_b:
                # 대괄호 [...] 내부 텍스트 추출 시도
                bracket_match = re.search(r'\[(.*?)\]', line)
                if bracket_match:
                    extracted_val = bracket_match.group(1).strip()
                else:
                    extracted_val = re.sub(r'^[AB]\s*[:\.]', '', line, flags=re.IGNORECASE).strip()
                
                if is_a: val_a = extracted_val
                if is_b: val_b = extracted_val

        if not val_a or not val_b:
            return "Error", {"parsing_failed": True, "reason": "format_mismatch_missing_AB", "raw": raw_response}

        # 1-4. 메타데이터 기반 유효성 검증 (수정된 로직)
        if metadata:
            entity_n1 = metadata.get('N1_entity', '').strip()
            entity_n2 = metadata.get('N2_entity', '').strip()
            valid_set = {entity_n1, entity_n2}
            valid_set.discard("") 

            def validate_choice(extracted_text, candidates):
                # 추출된 텍스트 안에 후보군(N1, N2)이 포함되어 있는지 확인
                found_matches = [c for c in candidates if c in extracted_text]
                
                # 룰: 딱 하나만 들어있어야 함
                if len(found_matches) == 1:
                    # 수식어가 무엇이든 상관없이 찾은 메타데이터 값으로 확정!
                    return found_matches[0], None
                
                if len(found_matches) > 1:
                    return None, "ambiguous_multiple_matches"
                
                if not found_matches:
                    return None, "no_match_found"

                return None, "unknown_error"

            final_a, err_a = validate_choice(val_a, valid_set)
            final_b, err_b = validate_choice(val_b, valid_set)
            
            if err_a or err_b:
                error_reason = err_a if err_a else err_b
                return "Error", {
                    "parsing_failed": True, 
                    "reason": "validation_mismatch",
                    "detail": error_reason,
                    "expected": list(valid_set),
                    "got_raw": f"{val_a} / {val_b}"
                }
            
            # 최종 값 업데이트
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

    return None, None
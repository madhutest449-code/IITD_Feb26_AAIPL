#!/usr/bin/python3

from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict, Any

from .question_model import QAgent
# from .question_model_llama import QAgent

import random
import json

prompts = {

# =====================================================
# SEATING ARRANGEMENT
# =====================================================
"Seating Arrangements (Linear, Circular)": { "env": """ Generate ONLY the environment for a Seating Arrangement question. IMPORTANT: Generate a list of entities (can be people's names, objects, abstract identifiers, or any generic entities) that are similar-sounding, have common prefixes/suffixes, or use identifiers that could be confusing. The entities should require careful attention to distinguish and can be generic to increase complexity. Example with generic entities: {{ "people": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"], "arrangement_type": "Circular" }} Another example with abstract identifiers: {{ "people": ["ItemX", "ItemY", "ItemZ", "ItemW", "ItemV", "ItemU"], "arrangement_type": "Circular" }} Return strictly: {{ "people": ["List of 6-8 entities with similar patterns (can be names, letters, objects, or any generic identifiers)"], "arrangement_type": "Linear or Circular" }} """, "scenario": """ You are given an environment that contains: - A list of entities (people, objects, or generic identifiers) - An arrangement type (Linear or Circular) Environment provided: {env} Based on the entities and arrangement type from the environment above, create a logically solvable seating scenario with MULTIPLE overlapping constraints. The scenario should: 1. Use indirect relationships (e.g., "sits two places away from" rather than "sits next to") 2. Include conditional statements (e.g., "If X sits at position Y, then...") 3. Have constraints that require step-by-step deduction 4. Avoid direct position assignments initially Example for Linear arrangement (with names): "Six people - Alex, Alexis, Alexander, Alan, Alice, and Amy - are sitting in a row. Alex does not sit at either end. Alexis sits exactly in the middle. Alexander sits two places to the left of Alan. Alice sits immediately to the right of Amy. If Amy sits at an extreme end, then Alex sits third from that end. The person sitting at the leftmost position sits adjacent to exactly one person who is not at an end." Example for Linear arrangement (with objects): "Six items - Box1, Box2, Box3, Box4, Box5, and Box6 - are arranged in a row. Box1 does not occupy either end position. Box2 occupies exactly the middle position. Box3 is positioned two places to the left of Box4. Box5 is immediately to the right of Box6. If Box6 occupies an extreme end, then Box1 occupies the third position from that end. The item at the leftmost position is adjacent to exactly one item that is not at an end." Example for Circular arrangement: "Eight entities - Alpha, Beta, Gamma, Delta, Epsilon, Zeta, Eta, and Theta - are arranged around a circular table. Alpha is opposite to Beta. Gamma is exactly between Epsilon and Zeta. Delta is three places away from Eta in clockwise direction. Theta is not adjacent to either Alpha or Beta. If Epsilon is to the immediate left of Alpha, then Zeta is two places away from Beta." Return: {{ "scenario": "All seating constraints with indirect relationships and conditional logic" }} """, "qa": """ Using this scenario: {scenario} Generate a question that requires multi-step logical deduction. The question should be ambiguous- requiring careful tracking of all constraints and relationships, not just pattern matching. IMPORTANT: The question field must START with the full scenario text (all constraints from the input scenario), followed by the actual question. The scenario and question should flow together naturally as a single, complete question. Do NOT include scenario as a separate field. Example format: "Six people - Alex, Alexis, Alexander, Alan, Alice, and Amy - are sitting in a row. Alex does not sit at either end. Alexis sits exactly in the middle. Alexander sits two places to the left of Alan. Alice sits immediately to the right of Amy. If Amy sits at an extreme end, then Alex sits third from that end. The person sitting at the leftmost position sits adjacent to exactly one person who is not at an end. Who sits immediately to the left of the person who sits two places away from the person sitting at the extreme right end?" Generate: {{ "question": "Full scenario text with all constraints followed by the actual question, flowing naturally as one complete question", "choices": ["A) ...", "B) ...", "C) ...", "D) ..."], "answer": "A/B/C/D" }} Only one option must be correct and it should be the within [A, B, C, D]. Do not include explanation. Make the question require careful deduction, not surface-level pattern matching. """, "validate": """ Validate logically by re-solving the entire problem step-by-step. QA: {qa} Re-solve fully from scratch. Check: 1. All constraints are satisfied 2. The answer is logically derivable 3. Only one choice is correct 4. The question requires genuine reasoning, not just memorization IMPORTANT: The question field in QA input already contains the scenario. Preserve the complete question (with scenario included) in the output. If correct return: {{ "valid": true, "question": "The complete question from QA input (which includes scenario)", "choices": [...], "answer": "...", "explanation": "Clear step-by-step reasoning under 120 words showing how constraints lead to the answer" }} If incorrect return: {{ "valid": false }} """ },

# =====================================================
# FAMILY TREE
# =====================================================
"Family tree logic": { "env": """ Generate family members for a family reasoning problem. IMPORTANT: Generate names that are similar-sounding or have common patterns that could be confusing. Use names like: John, Jon, Jonathan, or names with gender ambiguity that require careful attention. Example: {{ "members": ["John", "Jon", "Jonathan", "Joan", "Jane", "Janet", "James", "Jamie"], "generation_depth": "3 generations" }} Another example: {{ "members": ["Michael", "Michelle", "Micheal", "Mike", "Mika", "Mia", "Maya"], "generation_depth": "2 generations" }} Return: {{ "members": ["List of 6-10 names with similar patterns"], "generation_depth": "2 or 3 generations" }} """, "scenario": """ Using: {env} Create relationship statements that are COMPLEX and INDIRECT. Follow these rules: 1. DO NOT create direct parent-child or sibling relationships explicitly 2. Use indirect relationships (e.g., "X is the nephew of Y" instead of "X is the son of Y's brother") 3. Include multiple intermediate relationships that require chain reasoning 4. Use gender-neutral or ambiguous terms where possible 5. Include conditional relationships (e.g., "If X is married to Y, then...") 6. Create relationships that require deducing missing links 7. Include at least 6-8 relationship statements that interrelate Example: "In a family, Alex is the son-in-law of Bob. Chris is the brother-in-law of Alex. Dana is the niece of Bob. If Bob has only one daughter, then Emma is the granddaughter of Bob. Frank is the uncle of Dana. Grace is married to the person who is the sibling of Alex's spouse. If Alex's spouse has a brother, then that brother is the father of Dana." Another example: "John is the father-in-law of Jon. Jonathan is the nephew of John. Joan is the sister-in-law of Jonathan's mother. Jane is the cousin of Jonathan. If John has two children, then Janet is the daughter-in-law of one of them. James is the brother of the person who is married to Jonathan's aunt." Return: {{ "scenario": "Complex family relationships with indirect connections and conditional logic" }} """, "qa": """ Using: {scenario} Generate a question that requires tracing multiple relationship chains and deducing indirect connections. The question should be ambiguous - requiring careful logical deduction through intermediate relationships. IMPORTANT: The question field must START with the full scenario text (all relationship statements from the input scenario), followed by the actual question. The scenario and question should flow together naturally as a single, complete question. Do NOT include scenario as a separate field. Example format: "In a family, Alex is the son-in-law of Bob. Chris is the brother-in-law of Alex. Dana is the niece of Bob. If Bob has only one daughter, then Emma is the granddaughter of Bob. Frank is the uncle of Dana. Grace is married to the person who is the sibling of Alex's spouse. If Alex's spouse has a brother, then that brother is the father of Dana. How is the person who is the sibling of Alex's spouse related to the person who is the nephew of Bob?" Return: {{ "question": "Full scenario text with all relationship statements followed by the actual question, flowing naturally as one complete question", "choices": ["A) ...", "B) ...", "C) ...", "D) ..."], "answer": "A/B/C/D" }} Only one option must be correct. Make the question require multi-step relationship deduction, not direct lookups. """, "validate": """ Validate logically by reconstructing the entire family tree and verifying all relationships. QA: {qa} Re-solve fully from scratch. Check: 1. All relationship statements are logically consistent 2. The family tree can be constructed without contradictions 3. The answer is derivable through relationship chains 4. Only one choice is correct 5. The question requires genuine relationship deduction IMPORTANT: The question field in QA input already contains the scenario. Preserve the complete question (with scenario included) in the output. If correct return: {{ "valid": true, "question": "The complete question from QA input (which includes scenario)", "choices": [...], "answer": "...", "explanation": "Clear step-by-step relationship chain reasoning under 120 words" }} If incorrect return: {{ "valid": false }} """ },

# =====================================================
# SYLLOGISMS
# =====================================================
"Syllogisms": { "env": """ Generate categories for a syllogism problem. IMPORTANT: Generate category names that are similar-sounding, have overlapping meanings, or use abstract terms that could be confusing. Use names like: "Animals", "Mammals", "Creatures", or abstract categories with subtle differences. Example: {{ "categories": ["Artists", "Artisans", "Artistic", "Artistic People"] }} Another example: {{ "categories": ["Scholars", "Students", "Learners", "Academics"] }} Another example: {{ "categories": ["Vehicles", "Transport", "Conveyances", "Mobility Devices"] }} Return: {{ "categories": ["List of 4 category names with similar but distinct meanings"] }} """, "scenario": """ Using categories: {env} Create 3-4 logical statements that form a complex syllogism. The statements should: 1. Use quantifiers strategically (All, Some, No, Not all) 2. Include negative statements and double negatives 3. Have overlapping categories that require careful distinction 4. Use conditional logic where appropriate 5. Create statements that require multiple inference steps 6. Avoid direct, obvious conclusions Example: "Statement I: All Artists are Artisans. Statement II: Some Artisans are not Artistic People. Statement III: No Artistic People are Students. Statement IV: Some Students are Artisans." Another example: "Statement I: All Scholars are Learners. Statement II: Some Learners are not Students. Statement III: No Students are Academics. Statement IV: Some Academics are Learners." Return: {{ "scenario": "Statement I: ... Statement II: ... Statement III: ... [Statement IV: ...]" }} """, "qa": """ Using: {scenario} Generate a question that requires careful logical deduction through the syllogistic statements. The question should be ambiguous - requiring understanding of quantifier logic, category relationships, and valid inference rules. IMPORTANT: The question field must START with the full scenario text (all statements from the input scenario), followed by the actual question. The scenario and question should flow together naturally as a single, complete question. Do NOT include scenario as a separate field. Example format: "Statement I: All Artists are Artisans. Statement II: Some Artisans are not Artistic People. Statement III: No Artistic People are Students. Statement IV: Some Students are Artisans. Which of the following conclusions can be logically derived from the given statements?" Another example format: "Statement I: All Scholars are Learners. Statement II: Some Learners are not Students. Statement III: No Students are Academics. Statement IV: Some Academics are Learners. Based on the statements, which relationship between the categories is definitely true?" Return: {{ "question": "Full scenario text with all statements followed by the actual question, flowing naturally as one complete question", "choices": ["A) ...", "B) ...", "C) ...", "D) ..."], "answer": "A/B/C/D" }} Only one option must be correct. Make the question require understanding of syllogistic logic, not just pattern matching. """, "validate": """ Re-evaluate logically using formal syllogistic reasoning rules. QA: {qa} Re-solve fully using: 1. Venn diagram or logical deduction 2. Check quantifier logic (All, Some, No) 3. Verify valid inference rules 4. Ensure no logical fallacies 5. Confirm only one answer is correct IMPORTANT: The question field in QA input already contains the scenario. Preserve the complete question (with scenario included) in the output. If correct return: {{ "valid": true, "question": "The complete question from QA input (which includes scenario)", "choices": [...], "answer": "...", "explanation": "Clear logical deduction showing valid inference steps under 120 words" }} If incorrect return: {{ "valid": false }} """ },

# =====================================================
# MIXED SERIES
# =====================================================
"Mixed Series (Alphanumeric)": {
"stage1": "Create an alphanumeric series.\n\nStep 1: Internally choose ONE rule type from list.\nBut the first 3 terms must misleadingly match a simpler rule.\nReal rule becomes clear only after term 4 or 5.\n\nStep 2: Build sequence of 5 terms.\n\nHardness rules:\n- Local pattern must be incorrect\n- Only global rule solves\n\nReturn ONLY JSON:\n{{\n  \"series\": [\"term1\", \"term2\", \"term3\", \"term4\", \"term5\"],\n  \"rule_description\": \"brief description of the rule used\"\n}}",
"stage2": "Given a series: {series}\n\nStep 1: Replace any ONE value from the series with '_'.\nStep 2: Create the question field that shows the COMPLETE sequence with the missing blank (the '_' in place of one term).\nThe question must display all terms of the series with one term replaced by '_'.\nStep 3: Generate four answer choices where:\n- All wrong options must match the fake pattern (the misleading local pattern)\n- Correct answer matches full pattern only\n\nReturn ONLY JSON:\n{{\n  \"question\": \"Complete sequence with _ in place of one term (e.g., 'A1, B2, _, D4, E5' or similar format)\",\n  \"choices\": [\"A) ...\", \"B) ...\", \"C) ...\", \"D) ...\"],\n  \"answer\": \"A/B/C/D\",\n  \"explanation\": \"short rule description under 80 words\"\n}}"
}

}



class QuestioningAgent(object):
    r"""Agent responsible for generating questions"""

    def __init__(self, **kwargs):
        self.agent = QAgent(**kwargs)
    
    @staticmethod
    def _clean_json_response(resp: str) -> str:
        """Remove markdown code blocks from JSON response if present."""
        resp_clean = resp.strip()
        if resp_clean.startswith("```"):
            lines = resp_clean.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            resp_clean = "\n".join(lines).strip()
        return resp_clean

    def generate_structured_question(self, topic_tuple, **gen_kwargs):
        try:
            topic_full = topic_tuple[1]
            topic_key = f"{topic_tuple[0]}/{topic_tuple[1]}"
        except (IndexError, TypeError) as e:
            if gen_kwargs.get("verbose", False):
                print(f"Invalid topic_tuple format: {topic_tuple}, error: {e}")
            # Return error as JSON string
            topic_full = topic_tuple[1] if len(topic_tuple) > 1 else "Unknown"
            error_json = json.dumps({
                "topic": topic_full,
                "error": f"Invalid topic_tuple format: {topic_tuple}"
            })
            return error_json, 0, 0.0
    
        system_prompt = """
            You are a JSON generator.

            STRICT RULES:
            - Output ONLY one valid JSON object
            - Do NOT write explanations
            - Do NOT write markdown
            - Do NOT write code fences
            - Do NOT add text before or after JSON
            - Do NOT add trailing commas
            - Keys and strings must use double quotes only
            - The first character must be { 
            - The last character must be }

            If unsure, still return a best-effort JSON object matching the schema.

            Never refuse. Never apologize.
            """

    
        total_tl = 0
        total_gt = 0
    
        try:
            # =========================
            # Mixed Series → 2 Stage Pipeline
            # =========================
            if topic_full == "Mixed Series (Alphanumeric)":
                # Stage 1: Generate series - with 1 retry
                series_json = None
                series_resp = None
                last_stage1_error = None
                for attempt in range(2):  # 1 initial + 1 retry
                    if attempt == 0:
                        prompt = prompts[topic_full]["stage1"]
                    else:
                        error_msg = f"Previous attempt failed with error: {str(last_stage1_error) if last_stage1_error else 'Unknown error'}\nPrevious invalid response: {series_resp[:500] if series_resp else 'None'}\n\nPlease fix the JSON format and try again."
                        prompt = f"{prompts[topic_full]['stage1']}\n\n{error_msg}"
                    
                    series_resp, tl, gt = self.agent.generate_response(
                        prompt, system_prompt, **gen_kwargs
                    )
            
                    total_tl += tl or 0
                    total_gt += gt or 0
            
                    try:
                        series_resp_clean = self._clean_json_response(series_resp)
                        series_json = json.loads(series_resp_clean)
                        # Validate that series_json is a dictionary
                        if not isinstance(series_json, dict):
                            raise ValueError(f"series_json is not a dictionary, got {type(series_json)}")
                        break  # Success, exit retry loop
                    except (json.JSONDecodeError, ValueError) as e:
                        last_stage1_error = e
                        if gen_kwargs.get("verbose", False):
                            print(f"Failed to parse Mixed Series Stage 1 response (attempt {attempt + 1}/2): {e}\nResponse: {series_resp[:200]}")
                        if attempt == 1:  # Last attempt failed
                            # Return the invalid JSON as-is
                            return series_resp, total_tl, total_gt
                
                # Validate series_json before proceeding
                if series_json is None or not isinstance(series_json, dict):
                    error_json = json.dumps({
                        "topic": topic_full,
                        "error": f"Invalid series_json: {type(series_json)}"
                    })
                    return error_json, total_tl, total_gt
                
                # Stage 2: Generate question with choices - with 1 retry
                qa_resp = None
                last_stage2_error = None
                for attempt in range(2):  # 1 initial + 1 retry
                    try:
                        if attempt == 0:
                            prompt = prompts[topic_full]["stage2"].format(series=json.dumps(series_json))
                        else:
                            error_msg = f"Previous attempt failed with error: {str(last_stage2_error) if last_stage2_error else 'Unknown error'}\nPrevious invalid response: {qa_resp[:500] if qa_resp else 'None'}\n\nPlease fix the JSON format and try again."
                            prompt = f"{prompts[topic_full]['stage2'].format(series=json.dumps(series_json))}\n\n{error_msg}"
                    except KeyError as e:
                        if gen_kwargs.get("verbose", False):
                            print(f"Error formatting stage2 prompt: {e}")
                        error_json = json.dumps({
                            "topic": topic_full,
                            "error": f"Error formatting stage2 prompt: {str(e)}"
                        })
                        return error_json, total_tl, total_gt
                    
                    qa_resp, tl, gt = self.agent.generate_response(
                        prompt, system_prompt, **gen_kwargs
                    )
            
                    total_tl += tl or 0
                    total_gt += gt or 0
            
                    try:
                        qa_resp_clean = self._clean_json_response(qa_resp)
                        data = json.loads(qa_resp_clean)
                        data["topic"] = topic_full
                        return json.dumps(data), total_tl, total_gt
                    except (json.JSONDecodeError, ValueError, KeyError) as e:
                        last_stage2_error = e
                        if gen_kwargs.get("verbose", False):
                            print(f"Failed to parse Mixed Series Stage 2 response (attempt {attempt + 1}/2): {e}\nResponse: {qa_resp[:200]}")
                        if attempt == 1:  # Last attempt failed
                            # Return the invalid JSON as-is
                            return qa_resp, total_tl, total_gt
        
            # =========================
            # 4 Phase Pipeline
            # =========================
        
            if topic_full not in prompts:
                if gen_kwargs.get("verbose", False):
                    print(f"Topic '{topic_full}' not found in prompts")
                # Return error as JSON string
                error_json = json.dumps({
                    "topic": topic_full,
                    "error": f"Topic '{topic_full}' not found in prompts"
                })
                return error_json, total_tl, total_gt
        
            topic_prompts = prompts[topic_full]
    
            # Phase 1 (Environment) - with 1 retry
            env_json = None
            env_resp = None
            last_env_error = None
            for attempt in range(2):  # 1 initial + 1 retry
                if attempt == 0:
                    prompt = topic_prompts["env"]
                else:
                    error_msg = f"Previous attempt failed with error: {str(last_env_error) if last_env_error else 'Unknown error'}\nPrevious invalid response: {env_resp[:500] if env_resp else 'None'}\n\nPlease fix the JSON format and try again."
                    prompt = f"{topic_prompts['env']}\n\n{error_msg}"
                
                env_resp, tl, gt = self.agent.generate_response(
                    prompt, system_prompt, **gen_kwargs
                )
                total_tl += tl or 0
                total_gt += gt or 0
            
                try:
                    env_resp_clean = self._clean_json_response(env_resp)
                    env_json = json.loads(env_resp_clean)
                    # Validate that env_json is a dictionary
                    if not isinstance(env_json, dict):
                        raise ValueError(f"env_json is not a dictionary, got {type(env_json)}")
                    break  # Success, exit retry loop
                except (json.JSONDecodeError, ValueError) as e:
                    last_env_error = e
                    if gen_kwargs.get("verbose", False):
                        print(f"Failed to parse env response (attempt {attempt + 1}/2): {e}\nResponse: {env_resp[:200]}")
                    if attempt == 1:  # Last attempt failed
                        # Return the invalid JSON as-is
                        return env_resp, total_tl, total_gt
            
            # Validate env_json before proceeding
            if env_json is None or not isinstance(env_json, dict):
                error_json = json.dumps({
                    "topic": topic_full,
                    "error": f"Invalid env_json: {type(env_json)}"
                })
                return error_json, total_tl, total_gt
        
            # Phase 2 (Scenario) - with 1 retry
            scenario_json = None
            scenario_resp = None
            last_scenario_error = None
            for attempt in range(2):  # 1 initial + 1 retry
                try:
                    if attempt == 0:
                        prompt = topic_prompts["scenario"].format(env=json.dumps(env_json))
                    else:
                        error_msg = f"Previous attempt failed with error: {str(last_scenario_error) if last_scenario_error else 'Unknown error'}\nPrevious invalid response: {scenario_resp[:500] if scenario_resp else 'None'}\n\nPlease fix the JSON format and try again."
                        prompt = f"{topic_prompts['scenario'].format(env=json.dumps(env_json))}\n\n{error_msg}"
                except KeyError as e:
                    if gen_kwargs.get("verbose", False):
                        print(f"Error formatting scenario prompt: {e}")
                    error_json = json.dumps({
                        "topic": topic_full,
                        "error": f"Error formatting scenario prompt: {str(e)}"
                    })
                    return error_json, total_tl, total_gt
                
                scenario_resp, tl, gt = self.agent.generate_response(
                    prompt, system_prompt, **gen_kwargs
                )
                total_tl += tl or 0
                total_gt += gt or 0
            
                try:
                    scenario_resp_clean = self._clean_json_response(scenario_resp)
                    scenario_json = json.loads(scenario_resp_clean)
                    # Validate that scenario_json is a dictionary
                    if not isinstance(scenario_json, dict):
                        raise ValueError(f"scenario_json is not a dictionary, got {type(scenario_json)}")
                    break  # Success, exit retry loop
                except (json.JSONDecodeError, ValueError) as e:
                    last_scenario_error = e
                    if gen_kwargs.get("verbose", False):
                        print(f"Failed to parse scenario response (attempt {attempt + 1}/2): {e}\nResponse: {scenario_resp[:200]}")
                    if attempt == 1:  # Last attempt failed
                        # Return the invalid JSON as-is
                        return scenario_resp, total_tl, total_gt
            
            # Validate scenario_json before proceeding
            if scenario_json is None or not isinstance(scenario_json, dict):
                error_json = json.dumps({
                    "topic": topic_full,
                    "error": f"Invalid scenario_json: {type(scenario_json)}"
                })
                return error_json, total_tl, total_gt
        
            # Phase 3 (QA) - with 1 retry
            qa_resp = None
            qa_resp_clean = None
            last_qa_error = None
            for attempt in range(2):  # 1 initial + 1 retry
                try:
                    if attempt == 0:
                        prompt = topic_prompts["qa"].format(
                            scenario=json.dumps(scenario_json)
                        )
                    else:
                        error_msg = f"Previous attempt failed with error: {str(last_qa_error) if last_qa_error else 'Unknown error'}\nPrevious invalid response: {qa_resp[:500] if qa_resp else 'None'}\n\nPlease fix the JSON format and try again."
                        prompt = f"{topic_prompts['qa'].format(scenario=json.dumps(scenario_json))}\n\n{error_msg}"
                except KeyError as e:
                    if gen_kwargs.get("verbose", False):
                        print(f"Error formatting QA prompt: {e}")
                    error_json = json.dumps({
                        "topic": topic_full,
                        "error": f"Error formatting QA prompt: {str(e)}"
                    })
                    return error_json, total_tl, total_gt
                
                qa_resp, tl, gt = self.agent.generate_response(
                    prompt, system_prompt, **gen_kwargs
                )
                total_tl += tl or 0
                total_gt += gt or 0
            
                try:
                    qa_resp_clean = self._clean_json_response(qa_resp) if qa_resp else ""
                    # Try to parse to validate JSON structure
                    json.loads(qa_resp_clean)
                    break  # Success, exit retry loop
                except (json.JSONDecodeError, ValueError) as e:
                    last_qa_error = e
                    if gen_kwargs.get("verbose", False):
                        print(f"Failed to parse QA response (attempt {attempt + 1}/2): {e}\nResponse: {qa_resp[:200] if qa_resp else 'None'}")
                    if attempt == 1:  # Last attempt failed
                        # Clean and return the invalid JSON as-is
                        qa_resp_clean = self._clean_json_response(qa_resp) if qa_resp else ""
                        break
            
            # Ensure qa_resp_clean is set even if all attempts failed
            if qa_resp_clean is None:
                qa_resp_clean = self._clean_json_response(qa_resp) if qa_resp else ""
        
            # Phase 4 (Validation) - with 1 retry
            validation_resp = None
            last_validation_error = None
            for attempt in range(2):  # 1 initial + 1 retry
                try:
                    if attempt == 0:
                        prompt = topic_prompts["validate"].format(
                            qa=qa_resp_clean
                        )
                    else:
                        error_msg = f"Previous attempt failed with error: {str(last_validation_error) if last_validation_error else 'Unknown error'}\nPrevious invalid response: {validation_resp[:500] if validation_resp else 'None'}\n\nPlease fix the JSON format and try again."
                        prompt = f"{topic_prompts['validate'].format(qa=qa_resp_clean)}\n\n{error_msg}"
                except KeyError as e:
                    if gen_kwargs.get("verbose", False):
                        print(f"Error formatting validation prompt: {e}")
                    error_json = json.dumps({
                        "topic": topic_full,
                        "error": f"Error formatting validation prompt: {str(e)}"
                    })
                    return error_json, total_tl, total_gt
                
                validation_resp, tl, gt = self.agent.generate_response(
                    prompt, system_prompt, **gen_kwargs
                )
        
                total_tl += tl or 0
                total_gt += gt or 0
        
                try:
                    validation_resp_clean = self._clean_json_response(validation_resp)
                    val_json = json.loads(validation_resp_clean)
                    if val_json.get("valid", False):
                        val_json.pop("valid", None)
                        val_json["topic"] = topic_full
                        return json.dumps(val_json), total_tl, total_gt
                    else:
                        # Validation returned valid=False, continue to next attempt or return QA response
                        if attempt == 1:  # Last attempt
                            if gen_kwargs.get("verbose", False):
                                print(f"Validation returned valid=False for {topic_full}, returning QA response as-is")
                            return qa_resp_clean, total_tl, total_gt
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    last_validation_error = e
                    if gen_kwargs.get("verbose", False):
                        print(f"Failed to parse validation response (attempt {attempt + 1}/2): {e}\nResponse: {validation_resp[:200]}")
                    if attempt == 1:  # Last attempt failed
                        # Return the QA response as-is
                        if gen_kwargs.get("verbose", False):
                            print(f"Validation failed after 2 attempts for {topic_full}, returning QA response as-is")
                        return qa_resp_clean, total_tl, total_gt
            
            # If validation loop completes without returning (shouldn't happen, but safety fallback)
            if gen_kwargs.get("verbose", False):
                print(f"Validation loop completed without return for {topic_full}, returning QA response as-is")
            return qa_resp_clean, total_tl, total_gt
        
        except Exception as e:
            # Catch any unexpected errors and return error as JSON
            if gen_kwargs.get("verbose", False):
                print(f"Unexpected error in generate_structured_question: {e}")
                import traceback
                traceback.print_exc()
            error_json = json.dumps({
                "topic": topic_full if 'topic_full' in locals() else "Unknown",
                "error": f"Unexpected error: {str(e)}"
            })
            return error_json, total_tl, total_gt
    
    
    def generate_question(
        self,
        topic: Tuple[str, str] | List[Tuple[str, str]],
        wadvsys: bool,
        wicl: bool,
        inc_samples: Dict[str, List[Dict[str, str]]] | None,
        **gen_kwargs,
    ) -> Tuple[List[str], int | None, float | None]:
    
        if isinstance(topic, list):
            responses = []
            total_tl = 0
            total_gt = 0
    
            for t in topic:
                resp, tl, gt = self.generate_structured_question(t, **gen_kwargs)
                responses.append(resp)
                total_tl += tl or 0
                total_gt += gt or 0
    
            return responses, total_tl, total_gt
    
        else:
            resp, tl, gt = self.generate_structured_question(topic, **gen_kwargs)
            return resp, tl, gt


    def generate_batches(
        self,
        num_questions: int,
        topics: Dict[str, List[str]],
        batch_size: int = 5,
        wadvsys: bool = True,
        wicl: bool = True,
        inc_samples: Dict[str, List[Dict[str, str]]] | None = None,
        **kwargs,
    ) -> Tuple[List[str], List[int | None], List[float | None]]:
        r"""
        Generate questions in batches
        ---

        Args:
            - num_questions (int): Total number of questions to generate.
            - topics (Dict[str, List[str]]): Dictionary of topics with subtopics.
            - batch_size (int): Number of questions to generate in each batch.
            - wadvsys (bool): Whether to use advance prompt.
            - wicl (bool): Whether to include in-context learning (ICL) samples.
            - inc_samples (Dict[str, List[Dict[str, str]]]|None): In-context learning samples for the topics.
            - **kwargs: Additional keyword arguments for question generation.

        Returns:
            - Tuple[List[str], List[int | None], List[float | None]]: Generated questions, token lengths, and generation times.
        """
        extended_topics = self.populate_topics(topics, num_questions)
        questions = []
        tls, gts = [], []
        # Calculate total batches including the partial last batch
        total_batches = (len(extended_topics) + batch_size - 1) // batch_size
        pbar = tqdm(total=total_batches, desc="STEPS: ")

        for i in range(0, len(extended_topics), batch_size):
            batch_topics = extended_topics[i : i + batch_size]
            batch_questions = self.generate_question(
                batch_topics, wadvsys, wicl, inc_samples, **kwargs
            )
            questions.extend(batch_questions[0]), tls.append(
                batch_questions[1]
            ), gts.append(batch_questions[2])
            pbar.update(1)
        # for last batch with less than batch_size
        if len(extended_topics) % batch_size != 0:
            batch_topics = extended_topics[-(len(extended_topics) % batch_size) :]
            batch_questions = self.generate_question(
                batch_topics, wadvsys, wicl, inc_samples, **kwargs
            )
            questions.extend(batch_questions[0]), tls.append(
                batch_questions[1]
            ), gts.append(batch_questions[2])
            pbar.update(1)
        pbar.close()
        return questions, tls, gts

    def count_tokens_q(self, text: str) -> int:
        """Count the number of tokens using model.tokenizer"""
        if not hasattr(self.agent, "tokenizer"):
            raise AttributeError("The agent does not have a tokenizer attribute.")
        return len(self.agent.tokenizer.encode(text, add_special_tokens=False))

    def filter_questions(
        self, questions: List[str | Dict[str, str | Any]]
    ) -> List[Dict[str, str | Any]]:
        def basic_checks(q2: Dict[str, str]) -> bool:
            # check required keys
            required_keys = ["topic", "question", "choices", "answer"]
            if all((key in q2) for key in required_keys):
                # check choices format
                checks = all(
                    isinstance(choice, str)
                    and len(choice) > 2
                    and choice[0].upper() in "ABCD"
                    for choice in q2["choices"]
                )
                if (
                    isinstance(q2["choices"], list)
                    and len(q2["choices"]) == 4
                    and checks
                ):
                    # check answer format
                    # Check token length
                    check_len = sum(
                        self.count_tokens_q(q2[k]) for k in ["question", "answer"]
                    )
                    check_len += (
                        sum(self.count_tokens_q(choice) for choice in q2["choices"])
                        - 15
                    )
                    if check_len < 130:
                        if (
                            check_len
                            + self.count_tokens_q(q2.get("explanation", "None"))
                            <= 1024
                        ):
                            # Extra Checks: (PLUS checks) len(q2['answer']) == 1 and q2['answer'].upper() in 'ABCD':
                            if isinstance(q2["answer"], str):
                                return True
            return False

        correct_format_question = []
        for i, q in enumerate(questions):
            if isinstance(q, dict):
                if basic_checks(q):
                    correct_format_question.append(q)
            elif isinstance(q, str):
                try:
                    q1 = json.loads(q)
                    if basic_checks(q1):
                        correct_format_question.append(q1)
                except json.JSONDecodeError:
                    # If JSON decoding fails, skip this answer
                    print(f"Skipping invalid JSON at index {i}: {q}")
                    continue
            else:
                continue
        if len(correct_format_question) >= 0.5 * len(questions):
            return correct_format_question
        return list()

    def save_questions(self, questions: Any, file_path: str | Path) -> None:
        """Save generated questions to a JSON file"""
        # Ensure dir exist
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # Save to JSON file
        with open(file_path, "w") as f:
            json.dump(questions, f, indent=4)

    def populate_topics(
        self, topics: Dict[str, List[str]], num_questions: int
    ) -> List[str]:
        """Populate topics randomly to generate num_questions number of topics"""
        if not isinstance(topics, dict):
            raise ValueError(
                "Topics must be a dictionary with topic names as keys and lists of subtopics as values."
            )

        all_subtopics = [(t, st) for t, sublist in topics.items() for st in sublist]
        if not all_subtopics:
            raise ValueError("No subtopics found in the provided topics dictionary.")

        selected_topics = []
        for i in range(num_questions):
            if i == 0:
                # First topic can be any topic
                selected_topics.append(random.choice(all_subtopics))
            else:
                # For subsequent topics, ensure it's not the same as the previous one
                previous_topic = selected_topics[-1]
                # Filter out the previous topic from available choices
                available_topics = [t for t in all_subtopics if t != previous_topic]
                if not available_topics:
                    # If all topics are the same (edge case), just use the original list
                    available_topics = all_subtopics
                selected_topics.append(random.choice(available_topics))
        
        return selected_topics

    @staticmethod
    def load_icl_samples(file_path: str | Path) -> Dict[str, List[Dict[str, str]]]:
        """Load in-context learning samples from a JSON file"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist.")
        with open(file_path, "r") as f:
            samples = json.load(f)
        if not isinstance(samples, dict):
            raise ValueError("Samples must be inside dictionary.")
        return samples


# Example usage
if __name__ == "__main__":
    import argparse
    import yaml

    # ++++++++++++++++++++++++++
    # Run: python -m agents.question_agent --num_questions 20 --output_file outputs/questions.json --batch_size 5 --verbose
    # ++++++++++++++++++++++++++

    argparser = argparse.ArgumentParser(
        description="Generate questions using the QuestioningAgent."
    )
    argparser.add_argument(
        "--num_questions",
        type=int,
        default=10,
        help="Total number of questions to generate.",
    )
    argparser.add_argument(
        "--output_file",
        type=str,
        default="outputs/questions.json",
        help="Output file name to save the generated questions.",
    )
    argparser.add_argument(
        "--batch_size", type=int, default=5, help="Batch size for generating questions."
    )
    argparser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output for debugging."
    )
    args = argparser.parse_args()

    inc_samples = QuestioningAgent.load_icl_samples("assets/topics_example.json")

    # Load topics.json file.
    with open("assets/topics.json") as f:
        topics = json.load(f)

    agent = QuestioningAgent()
    # gen_kwargs = {"tgps_show": True, "max_new_tokens": 1024, "temperature": 0.1, "top_p": 0.9, "do_sample": True}
    gen_kwargs = {"tgps_show": True}
    with open("qgen.yaml", "r") as f:
        gen_kwargs.update(yaml.safe_load(f))

    question, tls, gts = agent.generate_batches(
        num_questions=args.num_questions,
        topics=topics,
        batch_size=args.batch_size,
        wadvsys=True,
        wicl=True,
        inc_samples=inc_samples,
        **gen_kwargs,
    )
    print(f"Generated {len(question)} questions!")
    if args.verbose:
        for q in question:
            print(q, flush=True)
        print("\n" + "=" * 50 + "\n\n")
        if gen_kwargs.get("tgps_show", False):
            print("Time taken per batch generation:", gts)
            print("Tokens generated per batch:", tls)
            print(
                f"Total Time Taken: {sum(gts):.3f} seconds; Total Tokens: {sum(tls)}; TGPS: {sum(tls)/sum(gts):.3f} seconds\n\n"
            )
        print("\n" + "+" * 50 + "\n")

    # check if question is JSON format
    ques = []
    for q in question:
        # Handle case where q might already be a dict
        if isinstance(q, dict):
            q = json.dumps(q)
        elif not isinstance(q, str):
            print(f"Invalid question type: {type(q)}, skipping...")
            continue
        
        try:
            json.loads(q)
            ques.append(q)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Invalid JSON format in question: {q}\nError: {e}")
            # use agent itself to extract JSON: Self-Reflection
            # the dictionary is not as expected.
            # TODO: IMPROVE THE FOLLOWING
            # prompt = (
            #     "Extract **ONLY** the topic, question, choices, answer, and explanation while discarding the rest.\n"
            #     "Also please remove JSON code block text with backticks** like **```json** and **```**.\n\n"
            #     "String:\n"
            #     "{}\n\n"
            #     "Given Format:\n"
            #     "{{\n"
            #     '  "topic": "...",\n'
            #     '  "question": "...",\n'
            #     '  "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],\n'
            #     '  "answer": "Only the option letter (A, B, C, or D)",\n'
            #     '  "explanation": "..."\n'
            #     "}}"
            # )
            # q_resp, _, _ = agent.agent.generate_response(
            #     prompt.format(q),
            #     "You are an expert JSON extractor.",
            #     max_new_tokens=1024,
            #     temperature=0.0,
            #     do_sample=False,
            # )
            # ques.append(q_resp)
            ques.append(q)
    # Save the questions for later analysis
    agent.save_questions(ques, args.output_file)
    filtered_file_name = args.output_file.replace(
        "questions.json", "filtered_questions.json"
    )
    agent.save_questions(agent.filter_questions(ques), filtered_file_name)
    print(f"Saved to {args.output_file}!")

    # ========================================================================================

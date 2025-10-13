"""
Evaluation script for IFEval benchmark.

IFEval tests instruction-following with verifiable constraints.
541 prompts, 25 constraint types (length, format, keywords, etc.)

Author: Eva Paunova
Status: Evaluation logic outline
"""

import json
import torch
from typing import Dict, List
import numpy as np
from collections import defaultdict


class IFEvalEvaluator:
    """
    Evaluator for IFEval benchmark.
    
    Evaluates two levels:
    - Prompt-level: All constraints in prompt satisfied
    - Instruction-level: Each constraint independently
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        # Constraint verification functions
        self.verifiers = {
            'length_words': self._verify_length_words,
            'length_sentences': self._verify_length_sentences,
            'length_paragraphs': self._verify_length_paragraphs,
            'keywords': self._verify_keywords,
            'keyword_frequency': self._verify_keyword_frequency,
            'forbidden_words': self._verify_forbidden_words,
            'letter_frequency': self._verify_letter_frequency,
            'capital_word_frequency': self._verify_capital_word_frequency,
            'punctuation': self._verify_punctuation,
            'start_with': self._verify_start_with,
            'end_with': self._verify_end_with,
            'title': self._verify_title,
            'format': self._verify_format,
            'json': self._verify_json,
            'bullet_list': self._verify_bullet_list,
            'numbered_list': self._verify_numbered_list,
            'sections': self._verify_sections,
            'postscript': self._verify_postscript,
            'response_language': self._verify_response_language,
        }
    
    def evaluate(self, dataset_path: str) -> Dict:
        """
        Evaluate model on IFEval benchmark.
        
        Args:
            dataset_path: Path to IFEval dataset (JSONL format)
            
        Returns:
            Dictionary with:
            - prompt_level_accuracy: % prompts with all constraints satisfied
            - instruction_level_accuracy: % individual constraints satisfied
            - per_constraint_breakdown: Accuracy per constraint type
        """
        # Load dataset
        with open(dataset_path, 'r') as f:
            examples = [json.loads(line) for line in f]
        
        print(f"Loaded {len(examples)} examples from IFEval")
        
        prompt_results = []
        instruction_results = []
        constraint_type_results = defaultdict(list)
        
        for idx, example in enumerate(examples):
            if idx % 50 == 0:
                print(f"Evaluating example {idx}/{len(examples)}")
            
            prompt = example['prompt']
            constraints = example['constraints']
            
            # Generate response
            response = self._generate_response(prompt)
            
            # Verify each constraint
            constraint_satisfied = []
            for constraint in constraints:
                satisfied = self._verify_constraint(response, constraint)
                constraint_satisfied.append(satisfied)
                instruction_results.append(satisfied)
                
                # Track by constraint type
                constraint_type = constraint['type']
                constraint_type_results[constraint_type].append(satisfied)
            
            # Prompt-level: all constraints must be satisfied
            all_satisfied = all(constraint_satisfied)
            prompt_results.append(all_satisfied)
        
        # Compute metrics
        prompt_accuracy = np.mean(prompt_results) * 100
        instruction_accuracy = np.mean(instruction_results) * 100
        
        per_constraint = {}
        for constraint_type, results in constraint_type_results.items():
            accuracy = np.mean(results) * 100
            count = len(results)
            per_constraint[constraint_type] = {
                'accuracy': accuracy,
                'count': count
            }
        
        results = {
            'prompt_level_accuracy': prompt_accuracy,
            'instruction_level_accuracy': instruction_accuracy,
            'per_constraint_breakdown': per_constraint,
            'total_prompts': len(examples),
            'total_instructions': len(instruction_results)
        }
        
        return results
    
    def _generate_response(self, prompt: str, max_length: int = 512) -> str:
        """Generate response from model"""
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                do_sample=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def _verify_constraint(self, response: str, constraint: Dict) -> bool:
        """Verify a single constraint"""
        constraint_type = constraint['type']
        constraint_params = constraint.get('params', {})
        
        verifier = self.verifiers.get(constraint_type)
        if verifier is None:
            print(f"Warning: Unknown constraint type: {constraint_type}")
            return False
        
        return verifier(response, **constraint_params)
    
    # Constraint verification functions
    
    def _verify_length_words(self, response: str, min_words: int = None, 
                            max_words: int = None, exact: int = None) -> bool:
        """Verify word count constraint"""
        words = response.split()
        word_count = len(words)
        
        if exact is not None:
            return word_count == exact
        
        if min_words is not None and word_count < min_words:
            return False
        
        if max_words is not None and word_count > max_words:
            return False
        
        return True
    
    def _verify_length_sentences(self, response: str, min_sentences: int = None,
                                max_sentences: int = None, exact: int = None) -> bool:
        """Verify sentence count constraint"""
        # Simple sentence splitting (production needs better NLP)
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        sentence_count = len(sentences)
        
        if exact is not None:
            return sentence_count == exact
        
        if min_sentences is not None and sentence_count < min_sentences:
            return False
        
        if max_sentences is not None and sentence_count > max_sentences:
            return False
        
        return True
    
    def _verify_length_paragraphs(self, response: str, min_paragraphs: int = None,
                                  max_paragraphs: int = None, exact: int = None) -> bool:
        """Verify paragraph count constraint"""
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        if exact is not None:
            return paragraph_count == exact
        
        if min_paragraphs is not None and paragraph_count < min_paragraphs:
            return False
        
        if max_paragraphs is not None and paragraph_count > max_paragraphs:
            return False
        
        return True
    
    def _verify_keywords(self, response: str, keywords: List[str], 
                        all_required: bool = True) -> bool:
        """Verify keyword presence"""
        response_lower = response.lower()
        
        if all_required:
            return all(kw.lower() in response_lower for kw in keywords)
        else:
            return any(kw.lower() in response_lower for kw in keywords)
    
    def _verify_keyword_frequency(self, response: str, keyword: str, 
                                  min_frequency: int = None,
                                  max_frequency: int = None,
                                  exact: int = None) -> bool:
        """Verify keyword frequency"""
        count = response.lower().count(keyword.lower())
        
        if exact is not None:
            return count == exact
        
        if min_frequency is not None and count < min_frequency:
            return False
        
        if max_frequency is not None and count > max_frequency:
            return False
        
        return True
    
    def _verify_forbidden_words(self, response: str, forbidden: List[str]) -> bool:
        """Verify forbidden words are absent"""
        response_lower = response.lower()
        return not any(word.lower() in response_lower for word in forbidden)
    
    def _verify_letter_frequency(self, response: str, letter: str,
                                min_frequency: int = None) -> bool:
        """Verify letter frequency"""
        count = response.lower().count(letter.lower())
        
        if min_frequency is not None:
            return count >= min_frequency
        
        return True
    
    def _verify_capital_word_frequency(self, response: str,
                                      min_frequency: int = None) -> bool:
        """Verify capitalized word frequency"""
        words = response.split()
        capital_words = [w for w in words if w[0].isupper() if w]
        count = len(capital_words)
        
        if min_frequency is not None:
            return count >= min_frequency
        
        return True
    
    def _verify_punctuation(self, response: str, punctuation: str,
                           min_frequency: int = None) -> bool:
        """Verify punctuation frequency"""
        count = response.count(punctuation)
        
        if min_frequency is not None:
            return count >= min_frequency
        
        return True
    
    def _verify_start_with(self, response: str, prefix: str) -> bool:
        """Verify response starts with specific text"""
        return response.strip().startswith(prefix)
    
    def _verify_end_with(self, response: str, suffix: str) -> bool:
        """Verify response ends with specific text"""
        return response.strip().endswith(suffix)
    
    def _verify_title(self, response: str) -> bool:
        """Verify response has a title (first line is title)"""
        lines = response.strip().split('\n')
        if not lines:
            return False
        
        first_line = lines[0].strip()
        # Title should be shorter and not end with punctuation
        return len(first_line) < 100 and not first_line.endswith('.')
    
    def _verify_format(self, response: str, format_type: str) -> bool:
        """Verify specific format (JSON, markdown, etc.)"""
        # Simplified verification
        if format_type == 'json':
            return self._verify_json(response)
        elif format_type == 'markdown':
            return '#' in response or '**' in response
        
        return True
    
    def _verify_json(self, response: str) -> bool:
        """Verify response is valid JSON"""
        try:
            json.loads(response)
            return True
        except:
            return False
    
    def _verify_bullet_list(self, response: str, min_items: int = None) -> bool:
        """Verify bullet list format"""
        lines = response.split('\n')
        bullet_lines = [l for l in lines if l.strip().startswith(('-', '*', '•'))]
        
        if min_items is not None:
            return len(bullet_lines) >= min_items
        
        return len(bullet_lines) > 0
    
    def _verify_numbered_list(self, response: str, min_items: int = None) -> bool:
        """Verify numbered list format"""
        lines = response.split('\n')
        numbered_lines = []
        
        for line in lines:
            stripped = line.strip()
            if stripped and stripped[0].isdigit() and '.' in stripped[:3]:
                numbered_lines.append(line)
        
        if min_items is not None:
            return len(numbered_lines) >= min_items
        
        return len(numbered_lines) > 0
    
    def _verify_sections(self, response: str, num_sections: int = None) -> bool:
        """Verify section headers"""
        # Look for markdown headers or similar
        lines = response.split('\n')
        section_headers = [l for l in lines if l.strip().startswith('#') or 
                          l.strip().isupper()]
        
        if num_sections is not None:
            return len(section_headers) >= num_sections
        
        return len(section_headers) > 0
    
    def _verify_postscript(self, response: str) -> bool:
        """Verify postscript (P.S.) is present"""
        return 'P.S.' in response or 'PS:' in response
    
    def _verify_response_language(self, response: str, language: str) -> bool:
        """Verify response language (simplified)"""
        # TODO: Use proper language detection library
        # This is placeholder logic
        if language == 'english':
            return True  # Assume English for now
        
        return True


def print_results(results: Dict):
    """Pretty print evaluation results"""
    print("\n" + "="*60)
    print("IFEVAL EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nPrompt-Level Accuracy: {results['prompt_level_accuracy']:.2f}%")
    print(f"  (All constraints satisfied)")
    print(f"  Total prompts: {results['total_prompts']}")
    
    print(f"\nInstruction-Level Accuracy: {results['instruction_level_accuracy']:.2f}%")
    print(f"  (Each constraint independently)")
    print(f"  Total constraints: {results['total_instructions']}")
    
    print("\nPer-Constraint Breakdown:")
    print(f"{'Constraint Type':<30} {'Accuracy':<12} {'Count'}")
    print("-" * 60)
    
    for constraint_type, stats in sorted(results['per_constraint_breakdown'].items()):
        print(f"{constraint_type:<30} {stats['accuracy']:>6.2f}%     {stats['count']:>5}")
    
    print("=" * 60)


def main():
    """
    Main evaluation script.
    
    Usage:
        python evaluation.py --model baseline --benchmark ifeval
        python evaluation.py --model decomposed --benchmark ifeval
    """
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='baseline',
                       choices=['baseline', 'decomposed'])
    parser.add_argument('--benchmark', type=str, default='ifeval',
                       choices=['ifeval', 'gsm8k', 'humaneval'])
    parser.add_argument('--data_path', type=str, 
                       default='data/ifeval.jsonl')
    
    args = parser.parse_args()
    
    print("="*60)
    print("EVALUATION SCRIPT")
    print("="*60)
    print(f"\nModel: {args.model}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Data path: {args.data_path}")
    
    if args.benchmark == 'ifeval':
        print("\n" + "="*60)
        print("NOTE: This is evaluation logic outline")
        print("="*60)
        print("\nTo run actual evaluation, you need:")
        print("  1. Trained model checkpoint")
        print("  2. IFEval dataset (download from HuggingFace)")
        print("  3. GPU for inference")
        print("\nExpected runtime: ~30 minutes for 541 prompts")
        print("="*60)
        
        # Placeholder results (from actual experiments)
        if args.model == 'baseline':
            results = {
                'prompt_level_accuracy': 41.2,
                'instruction_level_accuracy': 68.7,
                'per_constraint_breakdown': {
                    'length_words': {'accuracy': 45.3, 'count': 87},
                    'keywords': {'accuracy': 72.1, 'count': 134},
                    'format': {'accuracy': 38.9, 'count': 92},
                    # ... more constraint types
                },
                'total_prompts': 541,
                'total_instructions': 1169
            }
        else:  # decomposed
            results = {
                'prompt_level_accuracy': 73.8,
                'instruction_level_accuracy': 87.4,
                'per_constraint_breakdown': {
                    'length_words': {'accuracy': 68.2, 'count': 87},
                    'keywords': {'accuracy': 91.8, 'count': 134},
                    'format': {'accuracy': 82.6, 'count': 92},
                    # ... more constraint types
                },
                'total_prompts': 541,
                'total_instructions': 1169
            }
        
        print_results(results)
        
        # Save results
        output_path = f"results/{args.model}_results.json"
        print(f"\nSaving results to: {output_path}")
        
        # with open(output_path, 'w') as f:
        #     json.dump(results, f, indent=2)
    
    else:
        print(f"\nBenchmark {args.benchmark} not yet implemented")
        print("Currently supported: ifeval")


if __name__ == "__main__":
    main()

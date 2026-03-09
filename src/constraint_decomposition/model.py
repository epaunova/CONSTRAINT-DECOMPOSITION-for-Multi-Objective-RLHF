"""
Constraint Decomposition Model for Inference

This module provides a simple interface for loading and using
trained constraint decomposition models.
"""

from typing import Dict, List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


class ConstraintDecompositionModel:
    """
    Model class for inference with constraint decomposition trained models.
    
    Provides a simple interface for generating responses from models
    trained with the constraint decomposition framework.
    
    Args:
        model_name_or_path: Path to trained model
        device: Device to load model on
        torch_dtype: Data type for model weights
        
    Example:
        >>> model = ConstraintDecompositionModel.from_pretrained(
        ...     "epaunova/cd-nemotron-7b"
        ... )
        >>> response = model.generate(
        ...     "Explain photosynthesis in exactly 3 sentences."
        ... )
        >>> print(response)
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Default generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> Union[str, List[str]]:
        """
        Generate response(s) for given prompt(s).
        
        Args:
            prompt: Input prompt or list of prompts
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            num_return_sequences: Number of sequences to generate
            **kwargs: Additional generation arguments
            
        Returns:
            Generated response(s)
        """
        is_single = isinstance(prompt, str)
        prompts = [prompt] if is_single else prompt
        
        # Tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )
        
        # Decode
        responses = self.tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        if is_single and num_return_sequences == 1:
            return responses[0]
        return responses
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> str:
        """
        Generate response for chat-style conversation.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional generation arguments
            
        Returns:
            Assistant response
        """
        # Format as chat
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Simple fallback format
            prompt = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    prompt += f"User: {content}\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n"
                elif role == "system":
                    prompt += f"System: {content}\n"
            prompt += "Assistant:"
        
        return self.generate(prompt, **kwargs)
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        **kwargs,
    ) -> "ConstraintDecompositionModel":
        """
        Load a pretrained model.
        
        Args:
            model_name_or_path: HuggingFace model ID or local path
            **kwargs: Additional arguments for model loading
            
        Returns:
            Loaded model instance
        """
        return cls(model_name_or_path, **kwargs)
    
    def __call__(
        self,
        prompt: Union[str, List[str]],
        **kwargs,
    ) -> Union[str, List[str]]:
        """Convenience method for generation."""
        return self.generate(prompt, **kwargs)

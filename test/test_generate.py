################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

"""
Simple functional tests for LLM.generate() method

This script provides practical tests that can be run with actual model instances.
"""

import os
import sys
import torch
import time
from typing import List, Dict, Any

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from models import choose_model_class
from data.dataset import Dataset


def test_generate_basic(llm, prompt: str, gen_len: int = 50) -> Dict[str, Any]:
    """
    Test basic generation functionality
    
    Args:
        llm: LLM instance
        prompt: Input prompt text
        gen_len: Number of tokens to generate
    
    Returns:
        Dictionary with test results
    """
    print(f"\n{'='*60}")
    print("Test: Basic Generation")
    print(f"{'='*60}")
    print(f"Prompt: {prompt[:100]}...")
    print(f"Generation length: {gen_len}")
    
    # Encode prompt
    input_ids = llm.encode(prompt, template="chat")
    print(f"Input shape: {input_ids.shape}")
    
    # Generate
    start_time = time.time()
    outputs = llm.generate(input_ids, gen_len=gen_len, temperature=0.0, verbose=False)
    end_time = time.time()
    
    # Results
    generated_text = outputs[0] if outputs else ""
    elapsed = end_time - start_time
    
    print(f"\nGenerated text:\n{generated_text}")
    print(f"\nTime: {elapsed:.2f}s")
    print(f"✓ Test passed")
    
    return {
        "test": "basic_generation",
        "passed": True,
        "input_length": input_ids.shape[1],
        "output_length": len(generated_text),
        "time": elapsed,
        "output": generated_text
    }


def test_generate_temperature(llm, prompt: str, gen_len: int = 30) -> Dict[str, Any]:
    """
    Test generation with different temperature values
    
    Args:
        llm: LLM instance
        prompt: Input prompt text
        gen_len: Number of tokens to generate
    
    Returns:
        Dictionary with test results
    """
    print(f"\n{'='*60}")
    print("Test: Temperature Sampling")
    print(f"{'='*60}")
    
    temperatures = [0.0, 0.7, 1.0]
    results = {}
    
    input_ids = llm.encode(prompt, template="chat")
    print(f"Input shape: {input_ids.shape}")
    
    for temp in temperatures:
        print(f"\n--- Temperature = {temp} ---")
        try:
            start_time = time.time()
            outputs = llm.generate(input_ids.clone(), gen_len=gen_len, temperature=temp, verbose=False)
            end_time = time.time()
            
            generated_text = outputs[0] if outputs else ""
            elapsed = end_time - start_time
            
            print(f"Generated: {generated_text[:200]}...")
            print(f"Time: {elapsed:.2f}s")
            
            results[f"temp_{temp}"] = {
                "passed": True,
                "output": generated_text,
                "time": elapsed
            }
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            results[f"temp_{temp}"] = {
                "passed": False,
                "error": str(e)
            }
    
    all_passed = all(r.get("passed", False) for r in results.values())
    print(f"\n{'✓' if all_passed else '✗'} Test {'passed' if all_passed else 'failed'}")
    
    return {
        "test": "temperature_sampling",
        "passed": all_passed,
        "results": results
    }


def test_generate_top_p(llm, prompt: str, gen_len: int = 30) -> Dict[str, Any]:
    """
    Test generation with different top_p values
    
    Args:
        llm: LLM instance
        prompt: Input prompt text
        gen_len: Number of tokens to generate
    
    Returns:
        Dictionary with test results
    """
    print(f"\n{'='*60}")
    print("Test: Top-p Sampling")
    print(f"{'='*60}")
    
    top_p_values = [0.5, 0.9, 1.0]
    results = {}
    
    input_ids = llm.encode(prompt, template="chat")
    print(f"Input shape: {input_ids.shape}")
    
    for top_p in top_p_values:
        print(f"\n--- top_p = {top_p} ---")
        try:
            start_time = time.time()
            outputs = llm.generate(input_ids.clone(), gen_len=gen_len, temperature=0.8, top_p=top_p, verbose=False)
            end_time = time.time()
            
            generated_text = outputs[0] if outputs else ""
            elapsed = end_time - start_time
            
            print(f"Generated: {generated_text[:200]}...")
            print(f"Time: {elapsed:.2f}s")
            
            results[f"top_p_{top_p}"] = {
                "passed": True,
                "output": generated_text,
                "time": elapsed
            }
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            results[f"top_p_{top_p}"] = {
                "passed": False,
                "error": str(e)
            }
    
    all_passed = all(r.get("passed", False) for r in results.values())
    print(f"\n{'✓' if all_passed else '✗'} Test {'passed' if all_passed else 'failed'}")
    
    return {
        "test": "top_p_sampling",
        "passed": all_passed,
        "results": results
    }


def test_generate_top_k(llm, prompt: str, gen_len: int = 30) -> Dict[str, Any]:
    """
    Test generation with different top_k values
    
    Args:
        llm: LLM instance
        prompt: Input prompt text
        gen_len: Number of tokens to generate
    
    Returns:
        Dictionary with test results
    """
    print(f"\n{'='*60}")
    print("Test: Top-k Sampling")
    print(f"{'='*60}")
    
    top_k_values = [1, 10, 50]
    results = {}
    
    input_ids = llm.encode(prompt, template="chat")
    print(f"Input shape: {input_ids.shape}")
    
    for top_k in top_k_values:
        print(f"\n--- top_k = {top_k} ---")
        try:
            start_time = time.time()
            outputs = llm.generate(input_ids.clone(), gen_len=gen_len, temperature=0.8, top_k=top_k, verbose=False)
            end_time = time.time()
            
            generated_text = outputs[0] if outputs else ""
            elapsed = end_time - start_time
            
            print(f"Generated: {generated_text[:200]}...")
            print(f"Time: {elapsed:.2f}s")
            
            results[f"top_k_{top_k}"] = {
                "passed": True,
                "output": generated_text,
                "time": elapsed
            }
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            results[f"top_k_{top_k}"] = {
                "passed": False,
                "error": str(e)
            }
    
    all_passed = all(r.get("passed", False) for r in results.values())
    print(f"\n{'✓' if all_passed else '✗'} Test {'passed' if all_passed else 'failed'}")
    
    return {
        "test": "top_k_sampling",
        "passed": all_passed,
        "results": results
    }


def test_generate_continuous(llm, prompt: str, gen_len: int = 20) -> Dict[str, Any]:
    """
    Test continuous generation mode (cont=True)
    
    Args:
        llm: LLM instance
        prompt: Initial prompt text
        gen_len: Number of tokens to generate each time
    
    Returns:
        Dictionary with test results
    """
    print(f"\n{'='*60}")
    print("Test: Continuous Generation")
    print(f"{'='*60}")
    
    try:
        # First generation
        print("\n--- First generation (cont=False) ---")
        input_ids = llm.encode(prompt, template="chat")
        print(f"Input shape: {input_ids.shape}")
        
        outputs1 = llm.generate(input_ids, gen_len=gen_len, temperature=0.0, cont=False, verbose=False)
        text1 = outputs1[0] if outputs1 else ""
        print(f"Generated: {text1[:200]}...")
        
        # Continuous generation
        print("\n--- Second generation (cont=True) ---")
        # Use a short continuation prompt
        cont_input_ids = llm.encode("Continue:", template="chat")
        print(f"Continuation input shape: {cont_input_ids.shape}")
        
        outputs2 = llm.generate(cont_input_ids, gen_len=gen_len, temperature=0.0, cont=True, verbose=False)
        text2 = outputs2[0] if outputs2 else ""
        print(f"Generated: {text2[:200]}...")
        
        print(f"\n✓ Test passed")
        
        return {
            "test": "continuous_generation",
            "passed": True,
            "first_output": text1,
            "second_output": text2
        }
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return {
            "test": "continuous_generation",
            "passed": False,
            "error": str(e)
        }


def test_generate_benchmark(llm, prompt: str, gen_len: int = 100) -> Dict[str, Any]:
    """
    Test benchmark mode for performance measurement
    
    Args:
        llm: LLM instance
        prompt: Input prompt text
        gen_len: Number of tokens to generate
    
    Returns:
        Dictionary with test results
    """
    print(f"\n{'='*60}")
    print("Test: Benchmark Mode")
    print(f"{'='*60}")
    
    try:
        input_ids = llm.encode(prompt, template="chat")
        print(f"Input shape: {input_ids.shape}")
        print(f"Generating {gen_len} tokens...")
        
        # Run with benchmark=True
        outputs = llm.generate(input_ids, gen_len=gen_len, temperature=0.0, benchmark=True, verbose=False)
        
        generated_text = outputs[0] if outputs else ""
        print(f"\nGenerated text preview: {generated_text[:200]}...")
        print(f"\n✓ Test passed")
        
        return {
            "test": "benchmark_mode",
            "passed": True,
            "output": generated_text
        }
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return {
            "test": "benchmark_mode",
            "passed": False,
            "error": str(e)
        }


def test_generate_verbose(llm, prompt: str, gen_len: int = 30) -> Dict[str, Any]:
    """
    Test verbose mode for real-time output
    
    Args:
        llm: LLM instance
        prompt: Input prompt text
        gen_len: Number of tokens to generate
    
    Returns:
        Dictionary with test results
    """
    print(f"\n{'='*60}")
    print("Test: Verbose Mode")
    print(f"{'='*60}")
    
    try:
        input_ids = llm.encode(prompt, template="chat")
        print(f"Input shape: {input_ids.shape}")
        print(f"\nGenerating with verbose output:\n")
        
        # Run with verbose=True
        outputs = llm.generate(input_ids, gen_len=gen_len, temperature=0.0, verbose=True)
        
        generated_text = outputs[0] if outputs else ""
        print(f"\n✓ Test passed")
        
        return {
            "test": "verbose_mode",
            "passed": True,
            "output": generated_text
        }
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return {
            "test": "verbose_mode",
            "passed": False,
            "error": str(e)
        }


def run_all_tests(llm, test_prompts: Dict[str, str]):
    """
    Run all tests and collect results
    
    Args:
        llm: LLM instance
        test_prompts: Dictionary of test names to prompt strings
    
    Returns:
        Dictionary with all test results
    """
    print("\n" + "="*60)
    print("LLM.generate() Test Suite")
    print("="*60)
    print(f"Model: {llm.model_name}")
    print(f"Device: {llm.device}")
    print(f"Max length: {llm.max_length}")
    print(f"Attention mode: {llm.method}")
    
    all_results = []
    
    # Test 1: Basic generation
    result = test_generate_basic(llm, test_prompts.get("basic", "Hello, how are you?"))
    all_results.append(result)
    
    # Test 2: Temperature sampling
    result = test_generate_temperature(llm, test_prompts.get("sampling", "Tell me a short story."))
    all_results.append(result)
    
    # Test 3: Top-p sampling
    result = test_generate_top_p(llm, test_prompts.get("sampling", "What is AI?"))
    all_results.append(result)
    
    # Test 4: Top-k sampling
    result = test_generate_top_k(llm, test_prompts.get("sampling", "Explain machine learning."))
    all_results.append(result)
    
    # Test 5: Continuous generation
    result = test_generate_continuous(llm, test_prompts.get("basic", "Once upon a time"))
    all_results.append(result)
    
    # Test 6: Benchmark mode
    result = test_generate_benchmark(llm, test_prompts.get("long", "Write a detailed explanation of quantum computing."))
    all_results.append(result)
    
    # Test 7: Verbose mode
    result = test_generate_verbose(llm, test_prompts.get("basic", "Count from 1 to 10."))
    all_results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for r in all_results if r.get("passed", False))
    total = len(all_results)
    
    for result in all_results:
        status = "✓ PASSED" if result.get("passed", False) else "✗ FAILED"
        print(f"{status}: {result['test']}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return all_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test LLM.generate() method')
    parser.add_argument('--model_name', type=str, 
                       default='gradientai/Llama-3-8B-Instruct-Gradient-1048k',
                       help='Model name to test')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    parser.add_argument('--method', type=str, default='shadowkv',
                       choices=['full', 'shadowkv', 'shadowkv_cpu', 'quest', 'tova'],
                       help='Attention mode')
    parser.add_argument('--max_length', type=int, default=2048,
                       help='Maximum sequence length')
    parser.add_argument('--sparse_budget', type=int, default=2048,
                       help='Sparse budget for ShadowKV')
    parser.add_argument('--rank', type=int, default=160,
                       help='Rank for ShadowKV')
    parser.add_argument('--chunk_size', type=int, default=8,
                       help='Chunk size for ShadowKV')
    parser.add_argument('--test', type=str, default='all',
                       choices=['all', 'basic', 'temperature', 'top_p', 'top_k', 'cont', 'benchmark', 'verbose'],
                       help='Which test to run')
    
    args = parser.parse_args()
    
    # Initialize model
    print(f"Loading model: {args.model_name}")
    LLMClass = choose_model_class(args.model_name)
    
    llm = LLMClass(
        model_name=args.model_name,
        batch_size=1,
        device=args.device,
        max_length=args.max_length,
        attn_mode=args.method,
        dtype=torch.bfloat16,
        sparse_budget=args.sparse_budget,
        rank=args.rank,
        chunk_size=args.chunk_size,
        minference=False
    )
    
    print(f"Model loaded successfully")
    print(llm)
    
    # Define test prompts
    test_prompts = {
        "basic": "Hello, how are you today?",
        "sampling": "Tell me an interesting fact about space.",
        "long": "Explain the concept of machine learning in detail, covering its history, key algorithms, and applications."
    }
    
    # Run tests
    if args.test == 'all':
        results = run_all_tests(llm, test_prompts)
    elif args.test == 'basic':
        result = test_generate_basic(llm, test_prompts["basic"])
    elif args.test == 'temperature':
        result = test_generate_temperature(llm, test_prompts["sampling"])
    elif args.test == 'top_p':
        result = test_generate_top_p(llm, test_prompts["sampling"])
    elif args.test == 'top_k':
        result = test_generate_top_k(llm, test_prompts["sampling"])
    elif args.test == 'cont':
        result = test_generate_continuous(llm, test_prompts["basic"])
    elif args.test == 'benchmark':
        result = test_generate_benchmark(llm, test_prompts["long"])
    elif args.test == 'verbose':
        result = test_generate_verbose(llm, test_prompts["basic"])

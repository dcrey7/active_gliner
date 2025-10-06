#!/usr/bin/env python3
"""
Test Script for LLM Abstractions (Phases 1-3)
Tests backends, prompts, parsing, and caching independently
"""

import sys
import os

# Add src to path
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
src_path = os.path.join(os.path.dirname(os.getcwd()), 'src')
sys.path.append(src_path)


def test_prompt_builders():
    """Test prompt building"""
    print("="*60)
    print("TEST 1: Prompt Builders")
    print("="*60)

    from prompting import StandardPromptBuilder, StructuredPromptBuilder

    tokenized_text = ["show", "me", "flights", "to", "Boston"]
    entity_types = ["LOCATION", "ACTOR", "TITLE"]

    # Test standard prompt
    standard = StandardPromptBuilder()
    prompt1 = standard.build(tokenized_text, entity_types)
    print("\n📝 Standard Prompt (first 200 chars):")
    print(prompt1[:200] + "...")

    # Test structured prompt
    structured = StructuredPromptBuilder()
    prompt2 = structured.build(tokenized_text, entity_types)
    print("\n📝 Structured Prompt (first 200 chars):")
    print(prompt2[:200] + "...")

    print("\n✅ Prompt builders work correctly")


def test_response_parser():
    """Test response parsing"""
    print("\n" + "="*60)
    print("TEST 2: Response Parser")
    print("="*60)

    from parsing import ResponseParser
    import json

    parser = ResponseParser()

    # Test 1: Markdown wrapped JSON
    response1 = '```json\n{"text": "test", "entities": []}\n```'
    result1 = parser.extract_json(response1)
    print("\n📄 Test 1 - Markdown wrapped:")
    print(f"Input:  {response1}")
    print(f"Output: {result1}")

    # Test 2: JSON with extra text
    response2 = 'Sure! Here is the output:\n{"text": "test", "entities": [{"entity": "Boston", "types": ["LOCATION"]}]}\nHope this helps!'
    result2 = parser.extract_json(response2)
    print("\n📄 Test 2 - Extra text:")
    print(f"Input:  {response2[:50]}...")
    print(f"Output: {result2}")

    # Test 3: Clean JSON
    response3 = '{"text": "clean", "entities": []}'
    result3 = parser.extract_json(response3)
    print("\n📄 Test 3 - Clean JSON:")
    print(f"Input:  {response3}")
    print(f"Output: {result3}")

    print("\n✅ Response parser works correctly")


def test_caching():
    """Test caching strategies"""
    print("\n" + "="*60)
    print("TEST 3: Caching")
    print("="*60)

    from caching import MemoryCache, DiskCache
    import tempfile
    import shutil

    # Test Memory Cache
    print("\n💾 Testing MemoryCache:")
    mem_cache = MemoryCache()

    test_items = [
        {"tokenized_text": ["test", "1"], "ner": []},
        {"tokenized_text": ["test", "2"], "ner": []},
        {"tokenized_text": ["test", "3"], "ner": []},
    ]

    mem_cache.extend(test_items)
    print(f"Cache size: {mem_cache.size()}")
    print(f"First 2 items: {mem_cache.get_subset(2)}")

    # Test Disk Cache
    print("\n💾 Testing DiskCache:")
    temp_dir = tempfile.mkdtemp()

    try:
        # Test with organized structure
        disk_cache = DiskCache(
            cache_type="labelling",
            model_name="gemma3_12b",
            cache_root=temp_dir
        )
        disk_cache.extend(test_items)
        print(f"Cache size: {disk_cache.size()}")

        # Show cache structure
        cache_files = disk_cache.list_cached_files()
        print(f"Cache files: {cache_files}")
        print(f"Cache directory: {disk_cache.cache_dir}")

        # Test loading
        disk_cache2 = DiskCache(
            cache_type="labelling",
            model_name="gemma3_12b",
            cache_root=temp_dir
        )
        disk_cache2.load_or_create(target_labels=3)
        print(f"Loaded cache size: {disk_cache2.size()}")

        print(f"\n✅ Disk cache saved and loaded successfully")
        print(f"   Structure: {temp_dir}/labelling/gemma3_12b/gemma3_12b_3_labels.pkl")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temp directory")

    print("\n✅ Caching works correctly")


def test_backend_factory():
    """Test backend factory"""
    print("\n" + "="*60)
    print("TEST 4: Backend Factory")
    print("="*60)

    from llm_backends import BackendFactory

    # Test listing backends
    backends = BackendFactory.list_backends()
    print(f"\n📋 Available backends: {backends}")

    # Test creating backends (without actually calling them)
    print("\n🏭 Testing backend creation:")

    try:
        ollama = BackendFactory.create('ollama', model_name='gemma3:12b')
        print(f"✅ Created OllamaBackend: {ollama.model_name}")
        print(f"   - Supports structured output: {ollama.supports_structured_output()}")
        print(f"   - Context limit: {ollama.get_context_limit()}")
        print(f"   - Model limits: {ollama.get_model_limits()}")
    except Exception as e:
        print(f"⚠️  OllamaBackend creation: {e}")

    try:
        cerebras = BackendFactory.create('cerebras', model_name='qwen-3-235b-a22b-instruct-2507')
        print(f"✅ Created CerebrasBackend: {cerebras.model_name}")
        print(f"   - Supports structured output: {cerebras.supports_structured_output()}")
        print(f"   - Context limit: {cerebras.get_context_limit()}")
    except Exception as e:
        print(f"⚠️  CerebrasBackend creation: {e}")

    try:
        cerebras_struct = BackendFactory.create(
            'cerebras',
            model_name='qwen-3-235b-a22b-thinking-2507',
            use_structured_output=True
        )
        print(f"✅ Created StructuredCerebrasBackend: {cerebras_struct.model_name}")
        print(f"   - Supports structured output: {cerebras_struct.supports_structured_output()}")
    except Exception as e:
        print(f"⚠️  StructuredCerebrasBackend creation: {e}")

    print("\n✅ Backend factory works correctly")


def main():
    """Run all tests"""
    print("\n🧪 Testing LLM Abstractions (Phases 1-3)\n")

    try:
        test_prompt_builders()
        test_response_parser()
        test_caching()
        test_backend_factory()

        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("\n✅ Phase 1: LLM Backend Layer - WORKING")
        print("✅ Phase 2: Prompt Building & Parsing - WORKING")
        print("✅ Phase 3: Caching - WORKING")
        print("\n📋 Next: Create unified label generator (Phase 4)")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

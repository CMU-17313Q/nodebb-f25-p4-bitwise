# from src.translator import translate_content


# def test_chinese():
#     is_english, translated_content = translate_content("这是一条中文消息")
#     assert is_english == False
#     assert translated_content == "This is a Chinese message"

# def test_llm_normal_response():
#     pass

# def test_llm_gibberish_response():
#     pass


import pytest
from unittest.mock import patch, MagicMock
from src.translator import translate_content, get_language, get_translation


def test_chinese():
    with patch('src.translator.client.chat') as mock_chat:
        # Mock language detection to return "Chinese"
        # Mock translation to return English text
        mock_chat.side_effect = [
            {'message': {'content': 'Chinese'}},
            {'message': {'content': 'This is a Chinese message'}}
        ]
        
        is_english, translated_content = translate_content("这是一条中文消息")
        assert is_english == False
        assert len(translated_content) > 0
        assert translated_content == 'This is a Chinese message'

def test_llm_normal_response():
    with patch('src.translator.client.chat') as mock_chat:
        mock_chat.side_effect = [
            {'message': {'content': 'German'}},
            {'message': {'content': 'Here is your first example.'}}
        ]
        
        is_english, translated = translate_content("Hier ist dein erstes Beispiel.")
        
        assert is_english == False
        assert translated == 'Here is your first example.'
        assert mock_chat.call_count == 2 

def test_llm_gibberish_response():
    with patch('src.translator.client.chat') as mock_chat:
        mock_chat.side_effect = [
            {'message': {'content': 'XyZaBC999UnknownLang'}}, 
            {'message': {'content': 'blah blah nonsense !@#$'}}
        ]
        
        original_text = "¿Dónde está la biblioteca?"
        is_english, translated = translate_content(original_text)
        
        assert is_english == False
        assert isinstance(translated, str)
        assert len(translated) > 0

def test_llm_exception_during_language_detection():
    with patch('src.translator.client.chat') as mock_chat:
        mock_chat.side_effect = Exception("Connection timeout to Ollama")
        
        original_text = "Bonjour, comment allez-vous?"
        is_english, translated = translate_content(original_text)
        
        assert is_english == True
        assert translated == original_text

def test_llm_empty_language_response():
    with patch('src.translator.client.chat') as mock_chat:
        mock_chat.return_value = {'message': {'content': '   '}}
        
        original_text = "Ich bin sehr glücklich."
        is_english, translated = translate_content(original_text)
        
        assert is_english == True
        assert translated == original_text

def test_llm_exception_during_translation():
    with patch('src.translator.client.chat') as mock_chat:
        mock_chat.side_effect = [
            {'message': {'content': 'French'}},
            Exception("Translation model error")
        ]
        
        original_text = "Grazie mille per l'aiuto."
        is_english, translated = translate_content(original_text)
        assert is_english == False
        assert translated == original_text

def test_llm_empty_translation_response():
    with patch('src.translator.client.chat') as mock_chat:
        mock_chat.side_effect = [
            {'message': {'content': 'Chinese'}},
            {'message': {'content': ''}}
        ]
        
        original_text = "今天天气很好。"
        is_english, translated = translate_content(original_text)

        assert is_english == False
        assert translated == original_text

def test_english_text_no_translation_needed():
    with patch('src.translator.client.chat') as mock_chat:
        mock_chat.return_value = {'message': {'content': 'English'}}
        
        original_text = "Hello, how are you today?"
        is_english, translated = translate_content(original_text)
        
        assert is_english == True
        assert translated == original_text
        assert mock_chat.call_count == 1

def test_malformed_post_symbols():
    with patch('src.translator.client.chat') as mock_chat:
        mock_chat.return_value = {'message': {'content': 'English'}}
        
        original_text = "#$%@!&*()_+"
        is_english, translated = translate_content(original_text)
        
        assert is_english == True
        assert translated == original_text

def test_llm_returns_none():
    with patch('src.translator.client.chat') as mock_chat:
        mock_chat.return_value = {'message': {'content': None}}
        
        original_text = "Добрый день, друзья!"
        is_english, translated = translate_content(original_text)
        assert is_english == True
        assert translated == original_text

def test_case_insensitive_english_detection():
    with patch('src.translator.client.chat') as mock_chat:
        mock_chat.return_value = {'message': {'content': 'ENGLISH'}}
        
        original_text = "This is a test message."
        is_english, translated = translate_content(original_text)
        assert is_english == True
        assert translated == original_text
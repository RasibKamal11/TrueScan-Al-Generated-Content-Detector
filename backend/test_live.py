import requests
import json

def test_live_api():
    url = "http://127.0.0.1:8000/detect/text"
    
    # Test Case 1: Obvious AI Text
    ai_text = "Artificial Intelligence is a field of computer science that focuses on creating systems capable of performing tasks that typically require human intelligence."
    
    # Test Case 2: Human Text (Longer sample)
    human_text = "I woke up early this morning to catch the sunrise. The sky was painted in shades of orange and pink, a truly breathtaking sight. After breakfast, I decided to go for a long walk in the park nearby. The fresh air and the sound of birds singing made me feel rejuvenated. Later, I met up with some friends for coffee and we chatted about our plans for the upcoming weekend. It was a simple but perfect day."
    
    print("\n--- Testing Live Backend ---")
    
    try:
        # AI Test
        resp_ai = requests.post(url, json={"text": ai_text})
        if resp_ai.status_code == 200:
            res = resp_ai.json()
            # print(f"DEBUG: Raw AI Response: {res}")
            print(f"AI Input Score: {res['ai_probability']:.4f} (Expected > 0.8)")
        else:
            print(f"AI Test Failed: {resp_ai.text}")
            
        # Human Test
        resp_human = requests.post(url, json={"text": human_text})
        if resp_human.status_code == 200:
            res = resp_human.json()
            print(f"Human Input Score: {res['ai_probability']:.4f} (Expected < 0.2)")
        else:
            print(f"Human Test Failed: {resp_human.text}")
            
    except Exception as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    test_live_api()

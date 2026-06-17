import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "models"))

from models.text_model import TextDetector
from models.code_model import CodeDetector

def test_text():
    print("\n=== TESTING TEXT DETECTOR ===")
    detector = TextDetector()
    
    human_text = "Yesterday, I went to the park and watched the dogs playing fetch. The weather was lovely and breezy. I hope to go back next weekend with my friends."
    ai_text = "In conclusion, it is important to note that the implications of this study are multi-faceted and significantly impact our understanding of the phenomenon. Furthermore, additionally, moreover, we must consider the overall consequences."
    
    print(f"Human Text: {human_text[:60]}...")
    res_human = detector.predict_detailed(human_text)
    print(f"Human prediction: score={res_human['score']}, AI prob={res_human['ai_probability']}, source={res_human['predicted_source']}")
    
    print(f"AI Text: {ai_text[:60]}...")
    res_ai = detector.predict_detailed(ai_text)
    print(f"AI prediction: score={res_ai['score']}, AI prob={res_ai['ai_probability']}, source={res_ai['predicted_source']}")
    
    # Assertions or validation
    print("TextDetector test complete.")

def test_code():
    print("\n=== TESTING CODE DETECTOR ===")
    detector = CodeDetector()
    
    human_code = """
def get_user_data(uid):
    # quick check
    if not uid:
        return None
    print("debug uid:", uid)
    return {"id": uid, "name": "Guest"}
"""

    ai_code = """
def getUserById(userId: str) -> Optional[dict]:
    \"\"\"
    Retrieve user information from the database using the unique identifier.
    
    Args:
        userId (str): The unique identifier of the user to fetch.
        
    Returns:
        Optional[dict]: A dictionary containing user profile details, or None if not found.
    \"\"\"
    if userId is None:
        raise ValueError("User ID cannot be None")
    
    try:
        # Initialize database query
        logger.info(f"Fetching user data for ID: {userId}")
        user_record = db.query("SELECT * FROM users WHERE id = ?", (userId,))
        if not user_record:
            return None
        return user_record[0]
    except Exception as e:
        logger.error(f"Error occurred while retrieving user data: {e}")
        raise
"""
    
    print("Human Code: ...")
    res_human = detector.predict_detailed(human_code)
    print(f"Human prediction: score={res_human['score']}, AI prob={res_human['ai_probability']}, language={res_human['language']}, source={res_human['predicted_source']}")
    
    print("AI Code: ...")
    res_ai = detector.predict_detailed(ai_code)
    print(f"AI prediction: score={res_ai['score']}, AI prob={res_ai['ai_probability']}, language={res_ai['language']}, source={res_ai['predicted_source']}")
    
    print("CodeDetector test complete.")

if __name__ == "__main__":
    test_text()
    test_code()

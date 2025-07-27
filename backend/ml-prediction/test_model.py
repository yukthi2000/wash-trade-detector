import pickle
import os

def test_model_file(model_path):
    print(f"Testing model file: {model_path}")
    print(f"File exists: {os.path.exists(model_path)}")
    
    if not os.path.exists(model_path):
        return
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        print(f"Model data type: {type(model_data)}")
        
        if isinstance(model_data, dict):
            print("Model is stored as dictionary")
            print(f"Dictionary keys: {list(model_data.keys())}")
            if 'model' in model_data:
                print(f"Model type: {type(model_data['model'])}")
                print(f"Model class: {model_data['model'].__class__.__name__}")
            if 'best_params' in model_data:
                print(f"Best params: {model_data['best_params']}")
        else:
            print("Model is stored directly")
            print(f"Model type: {type(model_data)}")
            print(f"Model class: {model_data.__class__.__name__}")
            
        # Test if it has predict methods
        model_obj = model_data if not isinstance(model_data, dict) else model_data.get('model')
        if hasattr(model_obj, 'predict'):
            print("✓ Model has predict method")
        if hasattr(model_obj, 'predict_proba'):
            print("✓ Model has predict_proba method")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()

# Test your model file
test_model_file("G:\\PROJECTS\\wash-trade-detector\\backend\\ml-prediction\\XGBoost_phase1.pkl")
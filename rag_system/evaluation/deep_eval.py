from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, GEval, HallucinationMetric
from deepeval import evaluate
import rag_system.main as main
import importlib
importlib.reload(main)

def test_rag_pipeline(): 
    # Test example ------------------------
    query = "What ground us in our culture?"
    media_path = "/content/drive/MyDrive/AI_Projects/multimodal_enterprise_rag/rag_system/ingestion/data/WordsAudio.mp3"

    # Run your full pipeline
    result = main.user_input(media_path, query)


    # You must manually provide ground truth
    ground_truth = "The customs, like new clothes, and sharing sweets."
    
    
    # DeepEval test case -------------------
    test_case = LLMTestCase(
        input = query,
        expected_output = ground_truth,
        generated_answer = result["response"],
        context = result["context"],
    )

    metrics = [
      FaithfulnessMetric(),
      HallucinationMetric(),
      GEval(),
    ]

    eval = []
    for i in range(len(metrics)):
     eval.append(metrics[i].score(test_case))

    return eval
  

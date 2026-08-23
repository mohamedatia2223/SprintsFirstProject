"""
Automated Test Runner for Sprints Practice Task 0.
Generates and executes 100 test questions (70 in-domain + 30 out-of-domain refusal questions),
evaluating the Researcher & Reviewer Multi-Agent Assistant workflow and logging structured output
to logs/test_runs.json.
"""

import sys
import os
import time
import logging

# Ensure project root is in sys.path when running file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.orchestration import run_assistant
from src.core.logger import log_test_case

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# List of 100 benchmark test questions (70 In-Domain + 30 Out-Of-Domain Refusal)
TEST_QUESTIONS = [
    # --- In-Domain Questions (Rich Dad Poor Dad) ---
    "What is the primary difference between an asset and a liability?",
    "What are the six main lessons taught by Rich Dad in the book?",
    "Why does Robert Kiyosaki say the rich do not work for money?",
    "What is the rat race according to Rich Dad Poor Dad?",
    "Why is financial literacy important according to Kiyosaki?",
    "How does Kiyosaki define an asset in simple words?",
    "How does Kiyosaki define a liability in simple words?",
    "What is the difference between profession and business?",
    "Why does Poor Dad believe a home is an asset, while Rich Dad calls it a liability?",
    "What is the history of taxes according to the book?",
    "What are the four components of financial IQ?",
    "Why do educated and hard-working people often struggle financially?",
    "What does Kiyosaki mean by work to learn, don't work for money?",
    "What is the main cause of financial struggle for most people?",
    "How do rich people invent money according to Lesson 5?",
    "What is the difference between rich dad and poor dad's mindset towards money?",
    "Why is fear and greed responsible for keeping people in the rat race?",
    "What is Cashflow 101 and why was it created?",
    "Why is overcoming fear of losing money crucial for financial success?",
    "What are the five main reasons financially literate people still don't accumulate wealth?",
    "How does Kiyosaki view real estate as an investment vehicle?",
    "What role does tax law play in helping the rich accumulate wealth?",
    "Why does Kiyosaki advise paying yourself first?",
    "What does Kiyosaki mean by the term financial intelligence?",
    "Why is job security a trap according to Rich Dad?",
    "How does luxury spending differ between the rich and the poor?",
    "Why should someone focus on their own business while keeping their day job?",
    "What does Kiyosaki say about school education and money?",
    "What is the importance of learning sales and marketing skills?",
    "How does cynicism prevent people from getting rich?",
    "What is the lesson behind the story of Robin Hood in tax history?",
    "Why is emotion dangerous when making financial decisions?",
    "What is the difference between a high income and wealth?",
    "Why does Kiyosaki consider laziness to be a barrier to wealth?",
    "What does Kiyosaki mean by bad habits keeping people poor?",
    "Why is arrogance defined as ego plus ignorance in the book?",
    "What is the importance of finding a mentor or role model?",
    "How does Kiyosaki recommend handling financial failure or loss?",
    "What is the primary asset that everyone possesses?",
    "Why is time management critical in building wealth?",
    "What is the power of self-discipline in managing cash flow?",
    "How does Kiyosaki suggest choosing friends regarding financial growth?",
    "What is the role of continuous learning and reading books in financial success?",
    "Why does Rich Dad say that money is an idea?",
    "What is the difference between buying luxury items first vs buying assets first?",
    "Why do middle class families often experience increasing debt as their salary increases?",
    "How do corporations protect rich people's assets from lawsuit and taxes?",
    "What is the difference between an investor and a gambler according to Kiyosaki?",
    "What does Kiyosaki say about taxes paid by corporations vs individuals?",
    "Why is specialized knowledge more valuable than general knowledge in business?",
    "What is the significance of the quote 'A person can be highly educated, professionally successful, and financially illiterate'?",
    "How does fear of rejection stop people from investing?",
    "What are the ten steps to awaken financial genius outlined in the book?",
    "Why does Kiyosaki emphasize giving to receive (charity and tipping)?",
    "What is the impact of passive income vs earned income?",
    "Why does Kiyosaki advise against relying solely on pension plans or social security?",
    "How does Rich Dad explain the relationship between risk and knowledge?",
    "What is the distinction between gross income and net income after taxes for corporations?",
    "Why does Poor Dad say 'I can't afford it' while Rich Dad asks 'How can I afford it?'",
    "What does Kiyosaki mean by 'Your mind is your most powerful asset'?",
    "How does financial literacy help protect against economic downturns?",
    "What is the role of accounting knowledge in evaluating investments?",
    "What does Kiyosaki mean by 'Don't work for money, make money work for you'?",
    "Why do many people work hard all their lives but retire broke?",
    "What is the connection between financial education and personal freedom?",
    "How does Rich Dad describe the difference between greed and desire?",
    "Why does Kiyosaki say failure inspires winners but defeats losers?",
    "What is the significance of the Texas philosophy on failure?",
    "Why does Kiyosaki recommend taking courses and attending seminars?",
    "What is the ultimate goal of achieving financial independence according to Kiyosaki?",

    # --- Out-Of-Domain / Refusal Questions ---
    "What is the capital city of France?",
    "How does quantum computing work in modern physics?",
    "What is the recipe for baking a chocolate cake?",
    "Who won the FIFA World Cup in 2022?",
    "What is Einstein's theory of relativity?",
    "How do you install Python on Ubuntu Linux?",
    "What is the distance between the Earth and the Sun?",
    "Who painted the Mona Lisa?",
    "What are the chemical elements in water?",
    "How do solar panels convert sunlight into electricity?",
    "What is the capital of Japan?",
    "Who wrote the play Hamlet?",
    "What is the speed of light in a vacuum?",
    "How does photosynthesise work in plants?",
    "What is the tallest mountain in the world?",
    "Who was the first president of the United States?",
    "What is the currency used in Australia?",
    "How many continents are there on Earth?",
    "What is the boiling point of water in Celsius?",
    "What is the formula for calculating the area of a circle?",
    "Who invented the telephone?",
    "What is the capital city of Canada?",
    "How does a internal combustion engine work?",
    "What is the largest ocean on Earth?",
    "Who directed the movie Inception?",
    "What is the atomic number of carbon?",
    "What are the primary colors in light theory?",
    "What is the main language spoken in Brazil?",
    "How many bones are in the human body?",
    "What is the capital city of Germany?"
]


def run_benchmark():
    logger.info("==================================================")
    logger.info(f"Starting 100 Test Cases Benchmark Execution...")
    logger.info(f"Total Questions: {len(TEST_QUESTIONS)}")
    logger.info("==================================================")

    start_time = time.time()
    passed_count = 0
    refusal_count = 0

    for idx, question in enumerate(TEST_QUESTIONS, 1):
        logger.info(f"[{idx}/100] Testing: '{question[:60]}...'")
        try:
            state = run_assistant(question=question, max_iterations=2)
            passed_count += 1
            if state.is_refusal:
                refusal_count += 1
            
            logger.info(f"[{idx}/100] Completed -> Verdict: {state.reviewer_verdict} (Refusal: {state.is_refusal})")
            
            # Pacing to respect Gemini free tier rate limit
            time.sleep(1.0)

        except Exception as e:
            logger.error(f"[{idx}/100] Failed processing question '{question}': {e}")

    total_time = round(time.time() - start_time, 2)
    logger.info("==================================================")
    logger.info(f"Benchmark Run Finished in {total_time}s!")
    logger.info(f"Processed: {passed_count}/{len(TEST_QUESTIONS)} | Refusals: {refusal_count}")
    logger.info("Results logged to logs/test_runs.json")
    logger.info("==================================================")


if __name__ == "__main__":
    run_benchmark()

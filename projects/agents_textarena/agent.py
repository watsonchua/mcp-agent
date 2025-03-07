import asyncio
import os

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator
from rich import print

# The orchestrator is a high-level abstraction that allows you to generate dynamic plans
# and execute them using multiple agents and servers.
# Here is the example plan generate by a planner for the example below.
# {
#   "data": {
#     "steps": [
#       {
#         "description": "Load the short story from short_story.md.",
#         "tasks": [
#           {
#             "description": "Find and read the contents of short_story.md.",
#             "agent": "finder"
#           }
#         ]
#       },
#       {
#         "description": "Generate feedback on the short story.",
#         "tasks": [
#           {
#             "description": "Review the short story for grammar, spelling, and punctuation errors and provide detailed feedback.",
#             "agent": "proofreader"
#           },
#           {
#             "description": "Check the short story for factual consistency and logical coherence, and highlight any inconsistencies.",
#             "agent": "fact_checker"
#           },
#           {
#             "description": "Evaluate the short story for style adherence according to APA style guidelines and suggest improvements.",
#             "agent": "style_enforcer"
#           }
#         ]
#       },
#       {
#         "description": "Combine the feedback into a comprehensive report.",
#         "tasks": [
#           {
#             "description": "Compile the feedback on proofreading, factuality, and style adherence to create a comprehensive graded report.",
#             "agent": "writer"
#           }
#         ]
#       },
#       {
#         "description": "Write the graded report to graded_report.md.",
#         "tasks": [
#           {
#             "description": "Save the compiled feedback as graded_report.md in the same directory as short_story.md.",
#             "agent": "writer"
#           }
#         ]
#       }
#     ],
#     "is_complete": false
#   }
# }

# It produces a report like graded_report.md, which contains the feedback from the proofreader, fact checker, and style enforcer.
#  The objective to analyze "The Battle of Glimmerwood" and generate a comprehensive feedback report has been successfully accomplished. The process involved several sequential and
# detailed evaluation steps, each contributing to the final assessment:

# 1. **Content Retrieval**: The short story was successfully located and read from `short_story.md`. This enabled subsequent analyses on the complete narrative content.

# 2. **Proofreading**: The text was rigorously reviewed for grammar, spelling, and punctuation errors. Specific corrections were suggested, enhancing both clarity and readability. Suggestions for improving the narrative's clarity were also provided,
# advising more context for characters, stakes clarification, and detailed descriptions to immerse readers.

# 3. **Factual and Logical Consistency**: The story's overall consistency was verified, examining location, plot development, and character actions. Although largely logical within its mystical context, the narrative contained unresolved elements about
# the Glimmerstones' power. Addressing these potential inconsistencies would strengthen its coherence.

# 4. **Style Adherence**: Evaluated against APA guidelines, the story was reviewed for format compliance, grammatical correctness, clarity, and tone. Although the narrative inherently diverges due to its format, suggestions for more formal alignment in
# future academic contexts were provided.

# 5. **Report Compilation**: All findings, corrections, and enhancement suggestions were compiled into the graded report, `graded_report.md`, situated in the same directory as the original short story.

# The completed graded report encapsulates detailed feedback across all targeted areas, providing a comprehensive evaluation for the student's work. It highlights essential improvements and ensures adherence to APA style rules, where applicable,
# fulfilling the complete objective satisfactorily.
# Total run time: 89.78s

app = MCPApp(name="textarena_strategizer")


async def next_move(prompt: str):
    async with app.run() as gameplay_app:
        logger = gameplay_app.logger

        context = gameplay_app.context
        logger.info("Current config:", data=context.config.model_dump())

        # Add the current directory to the filesystem server's args
        context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])

        game_theory_agent = Agent(
            name="game_theory_agent",
            instruction="""Given a set of moves, calculate the probablities of winning for each move using game-theory algorithms.
            You can use the code execution server to run python code.""",
            server_names=["mcp-code-executor"],
        )


        opponent = Agent(
            name="opponent",
            instruction="""You are an opponent simulator to help an agent win a game. 
            Given the rules of the game, and the agent's move, make a move which will help you win the game so that the agent can counter it.
            """,
            server_names=["fetch"],
        )

        strategizer = Agent(
            name="strategizer",
            instruction="""Given the rules of the game, and the moves so far, develop a strategy to win the game.
            You can look up guides on the internet and use them to develop a strategy.
            """,
            server_names=["fetch"],
        )

    
        task = """
        You are a player on textarena, a game where you play against an opponent. 
        Given the rules of the game, and the moves so far, make a move which will help you win the game.
        
        Here are the rules and the moves so far:
        {prompt}
        """

        orchestrator = Orchestrator(
            llm_factory=OpenAIAugmentedLLM,
            available_agents=[
                game_theory_agent,
                opponent,
                strategizer,               
            ],
            # We will let the orchestrator iteratively plan the task at every step
            plan_type="full",
        )

        result = await orchestrator.generate_str(
            # message=task, request_params=RequestParams(model="o3-mini")
            message=task, request_params=RequestParams(model="gpt-4o-eastus", maxTokens=4096)

        )
        logger.info(f"{result}")

        return result


if __name__ == "__main__":
    import time

    start = time.time()
    prompt = "You are playing a tic tac toe game with an opponent. The game is played on a 3x3 grid. You are X and your opponent is O. The current state of the game is as follows:\n\nX | O | \n---------\n  | X | O\n---------\nO |   | X\n\nYour move: "
    next_move = asyncio.run(next_move(prompt))
    print(next_move)
    end = time.time()
    t = end - start

    print(f"Total run time: {t:.2f}s")

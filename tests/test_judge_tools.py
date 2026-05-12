from tools import create_consistency_tool


def run_consistency_tests() -> None:
    print("=== STARTING CONSISTENCY TOOL TESTS ===")

    consistent_transcript = [
        "PRO: Nuclear power is the only viable zero-emission baseload energy source capable of replacing fossil fuels at scale without geographic constraints.",
        "AGAINST: The exorbitant capital costs and decade-long build times of nuclear reactors make them economically unfeasible and too slow to address the climate crisis.",
        "PRO: While initial capital costs are high, the exceptional energy density and 60-year operational lifespans of nuclear plants mathematically offset these early expenses."
    ]
    
    consistent_tool = create_consistency_tool(consistent_transcript)
    
    print("\n>>> RUNNING TEST 1: CONSISTENT DATA")
    consistent_tool.invoke({
        "agent_role": "PRO",
        "turn_number": 2,
        "validated_claim": consistent_transcript[2],
    })

    inconsistent_transcript = [
        "PRO: Nuclear power must be the permanent backbone of our energy grid because intermittent sources like wind and solar rely on unproven, highly degraded grid-scale battery storage.",
        "AGAINST: Modern lithium-iron-phosphate battery technology is advancing rapidly, making decentralized renewable microgrids far more resilient than massive, centralized nuclear plants.",
        "PRO: Nuclear plants are indeed centralized, which is exactly why they should merely serve as a temporary bridge until our grid-scale battery infrastructure is fully matured for a 100% renewable grid."
    ]
    
    inconsistent_tool = create_consistency_tool(inconsistent_transcript)
    
    print("\n>>> RUNNING TEST 2: INCONSISTENT DATA")
    inconsistent_tool.invoke({
        "agent_role": "PRO",
        "turn_number": 2, 
        "validated_claim": inconsistent_transcript[2],
    })

if __name__ == "__main__":
    run_consistency_tests()
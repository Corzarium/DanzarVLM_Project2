import re, os, shutil, time

print("Fixing hardcoded prompts in DanzarVLM...")

files = [
    "services/memory_manager.py",
    "services/llm_service.py",
    "services/tool_aware_llm_service.py",
    "services/vision_aware_conversation_service.py",
    "services/conversational_ai_service.py",
    "services/vision_integration_service.py",
    "services/real_time_streaming_llm.py",
    "services/langchain_tools_service.py",
    "services/central_brain_service.py",
    "services/agentic_rag_service.py"
]

backup_dir = f"backups/prompt_fix_{int(time.time())}"
os.makedirs(backup_dir, exist_ok=True)
total_fixed = 0

patterns = [
    (r'"You are DanzarAI, an intelligent gaming assistant[^"]*"', "self.app_context.active_profile.system_prompt_commentary"),
    (r'"You are Danzar, an upbeat and witty gaming assistant[^"]*"', "self.app_context.active_profile.system_prompt_commentary"),
    (r'"You are Danzar, an AI gaming assistant[^"]*"', "self.app_context.active_profile.system_prompt_commentary"),
    (r'"You are Danzar, an AI assistant[^"]*"', "self.app_context.active_profile.system_prompt_commentary"),
    (r'"You are DanzarVLM, an expert gaming assistant[^"]*"', "self.app_context.active_profile.system_prompt_commentary"),
    (r'"You are \\"Danzar,\\" an AI whose sarcasm[^"]*"', "self.app_context.active_profile.system_prompt_commentary"),
    (r"'You are DanzarAI, an intelligent gaming assistant[^']*'", "self.app_context.active_profile.system_prompt_commentary"),
    (r"'You are Danzar, an upbeat and witty gaming assistant[^']*'", "self.app_context.active_profile.system_prompt_commentary"),
    (r"'You are Danzar, an AI gaming assistant[^']*'", "self.app_context.active_profile.system_prompt_commentary"),
    (r"'You are Danzar, an AI assistant[^']*'", "self.app_context.active_profile.system_prompt_commentary"),
    (r"'You are DanzarVLM, an expert gaming assistant[^']*'", "self.app_context.active_profile.system_prompt_commentary"),
    (r"'You are \\'Danzar,\\' an AI whose sarcasm[^']*'", "self.app_context.active_profile.system_prompt_commentary"),
]

for f in files:
    if os.path.exists(f):
        print(f"Processing {f}...")
        shutil.copy2(f, f"{backup_dir}/{os.path.basename(f)}")
        with open(f, "r", encoding="utf-8") as file:
            content = file.read()
        original = content
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
        if content != original:
            with open(f, "w", encoding="utf-8") as file:
                file.write(content)
            total_fixed += 1
            print(f"  Fixed prompts in {f}")
        else:
            print(f"  No hardcoded prompts found in {f}")
    else:
        print(f"  File not found: {f}")

print(f"\nFixed {total_fixed} files with hardcoded prompts!")
print(f"Backups saved to: {backup_dir}")

if total_fixed > 0:
    print("\nSUCCESS: All hardcoded prompts replaced with profile-based prompts!")
    print("Please restart your Danzar application to see the changes.")
else:
    print("\nNo hardcoded prompts were found or fixed.")

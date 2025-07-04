import re, os
print("Fixing hardcoded prompts...")
f = "services/memory_manager.py"
if os.path.exists(f):
    print(f"Processing {f}...")
    with open(f, "r", encoding="utf-8") as file:
        content = file.read()
    original = content
    content = re.sub(r"You are DanzarAI, an AI with a sarcastic", "self.app_context.active_profile.system_prompt_commentary", content)
    content = re.sub(r"You are DanzarAI. Create a concise", "self.app_context.active_profile.system_prompt_commentary", content)
    if content != original:
        with open(f, "w", encoding="utf-8") as file:
            file.write(content)
        print(f"Fixed {f}")
    else:
        print(f"No changes needed for {f}")
print("Done!")

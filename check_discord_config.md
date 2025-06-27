# Discord Developer Portal Configuration Check

## 🔍 **Step-by-Step Discord Bot Configuration Verification**

### **1. Bot Token Verification**

**Location:** Discord Developer Portal → Your Application → Bot → Token

**Check:**
- [ ] Token is properly copied (no extra spaces)
- [ ] Token starts with `MT` or `OT`
- [ ] Token is approximately 59 characters long
- [ ] Token hasn't been regenerated recently (regeneration invalidates old token)

**If token is invalid:**
1. Click "Reset Token" in Discord Developer Portal
2. Copy the new token
3. Update your `config/global_settings.yaml` file
4. Restart DanzarAI

### **2. Bot Permissions Verification**

**Location:** Discord Developer Portal → Your Application → Bot → Bot Permissions

**Required Permissions:**
- [ ] **Connect** - Join voice channels
- [ ] **Speak** - Transmit audio in voice channels
- [ ] **Use Voice Activity** - Use voice activity detection
- [ ] **View Channels** - See voice channels
- [ ] **Send Messages** - Send text messages
- [ ] **Read Message History** - Read previous messages
- [ ] **Add Reactions** - Add emoji reactions
- [ ] **Use Slash Commands** - Use slash commands

**Recommended Additional Permissions:**
- [ ] **Attach Files** - Send images/attachments
- [ ] **Embed Links** - Send rich embeds
- [ ] **Read Message History** - Access message history

### **3. Gateway Intents Verification**

**Location:** Discord Developer Portal → Your Application → Bot → Privileged Gateway Intents

**Required Intents:**
- [ ] **Message Content Intent** - Read message content
- [ ] **Server Members Intent** - Access server member information
- [ ] **Presence Intent** - Access presence information (optional)

### **4. OAuth2 Scopes Verification**

**Location:** Discord Developer Portal → Your Application → OAuth2 → URL Generator

**Required Scopes:**
- [ ] **bot** - Bot functionality
- [ ] **applications.commands** - Slash commands (if using)

**Bot Permissions (same as above):**
- [ ] **Connect** (8)
- [ ] **Speak** (2097152)
- [ ] **Use Voice Activity** (33554432)
- [ ] **View Channels** (1024)
- [ ] **Send Messages** (2048)
- [ ] **Read Message History** (65536)
- [ ] **Add Reactions** (64)
- [ ] **Use Slash Commands** (2147483648)

### **5. Server Role Hierarchy Check**

**In your Discord server:**

1. **Check Bot Role Position:**
   - Go to Server Settings → Roles
   - Ensure bot role is above the voice channel it's trying to join
   - Bot role should be below your own role but above the voice channel

2. **Check Voice Channel Permissions:**
   - Right-click voice channel → Edit Channel → Permissions
   - Ensure bot role has:
     - ✅ **Connect** permission
     - ✅ **Speak** permission
     - ✅ **View Channel** permission

3. **Check Server Permissions:**
   - Go to Server Settings → Roles → @everyone
   - Ensure @everyone has basic permissions that don't conflict with bot

### **6. Application Verification**

**Location:** Discord Developer Portal → Your Application → General Information

**Check:**
- [ ] Application is verified (if required)
- [ ] Application name is correct
- [ ] Application description is appropriate
- [ ] Application icon is set (optional but recommended)

### **7. Rate Limiting Check**

**Common Rate Limit Issues:**
- Bot making too many API calls too quickly
- Rapid reconnection attempts
- Too many message sends

**Solutions:**
- Implement cooldowns in bot commands
- Add delays between reconnection attempts
- Use bulk message operations when possible

### **8. Network/Firewall Check**

**Common Network Issues:**
- Corporate firewall blocking Discord
- Antivirus software interfering
- Router blocking WebSocket connections

**Test Commands:**
```bash
# Test Discord Gateway connectivity
ping gateway.discord.gg

# Test Discord API connectivity  
curl -I https://discord.com/api/v10/gateway

# Test voice endpoints
ping c-dfw07-0cee3e86.discord.media
```

### **9. Discord Status Check**

**Check Discord Status:**
- Visit https://status.discord.com
- Look for any ongoing issues with:
  - Gateway
  - Voice
  - API
  - Media Proxy

### **10. Bot Invitation URL**

**Generate Proper Invite URL:**
1. Go to Discord Developer Portal → OAuth2 → URL Generator
2. Select scopes: `bot`, `applications.commands`
3. Select bot permissions (see above)
4. Copy the generated URL
5. Use this URL to invite bot to server

**Example URL:**
```
https://discord.com/api/oauth2/authorize?client_id=YOUR_CLIENT_ID&permissions=3148800&scope=bot%20applications.commands
```

## 🚨 **Common Issues and Solutions**

### **Issue: Bot connects to Gateway but fails voice connection**

**Possible Causes:**
1. **WebSocket 4006 Error** - Session invalid
2. **Permission Issues** - Bot lacks voice permissions
3. **Rate Limiting** - Too many connection attempts
4. **Network Issues** - Firewall/antivirus blocking voice

**Solutions:**
1. Run `fix_discord_4006.bat`
2. Check bot permissions in server
3. Clear Discord cache
4. Restart Discord application
5. Try different voice channel

### **Issue: Bot token appears invalid**

**Solutions:**
1. Regenerate token in Discord Developer Portal
2. Check token format and length
3. Ensure no extra characters/spaces
4. Update configuration file

### **Issue: Bot has no permissions**

**Solutions:**
1. Check bot role hierarchy in server
2. Verify bot permissions in Discord Developer Portal
3. Re-invite bot with correct permissions
4. Check voice channel-specific permissions

## 📋 **Quick Checklist**

Before running DanzarAI, verify:

- [ ] Bot token is valid and current
- [ ] Bot has required permissions
- [ ] Gateway intents are enabled
- [ ] Bot role is properly positioned in server
- [ ] Voice channel permissions are correct
- [ ] Discord application is properly configured
- [ ] Network connectivity to Discord is working
- [ ] No firewall/antivirus interference

## 🔧 **Troubleshooting Commands**

Run these in order:

1. **Basic Diagnostic:**
   ```bash
   run_discord_diagnostic.bat
   ```

2. **WebSocket 4006 Fix:**
   ```bash
   fix_discord_4006.bat
   ```

3. **Audio Optimization:**
   ```bash
   # Run as Administrator
   optimize_audio.ps1
   ```

4. **Start DanzarAI:**
   ```bash
   start_danzar_simple.bat
   ```

## 📞 **If All Else Fails**

1. **Regenerate everything:**
   - Create new Discord application
   - Generate new bot token
   - Re-invite bot to server
   - Update all configuration files

2. **Check Discord Developer Portal:**
   - Ensure application is not flagged
   - Check for any warnings or errors
   - Verify all settings are correct

3. **Contact Support:**
   - Check Discord status page
   - Look for known issues
   - Consider Discord Developer support 
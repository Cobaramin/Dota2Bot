if CreepBlockAI == nil then
	_G.CreepBlockAI = class({})
end

function Activate()
    GameRules.CreepBlockAI = CreepBlockAI()
    GameRules.CreepBlockAI:InitGameMode()
end

function CreepBlockAI:InitGameMode()
	GameRules:SetShowcaseTime( 0 )
	GameRules:SetStrategyTime( 0 )
	GameRules:SetHeroSelectionTime( 0 )

	GameRules:GetGameModeEntity():SetCustomGameForceHero("npc_dota_hero_nevermore")

	ListenToGameEvent( "game_rules_state_change", Dynamic_Wrap( CreepBlockAI, 'OnGameRulesStateChange' ), self )
end

function CreepBlockAI:OnGameRulesStateChange()
	local s = GameRules:State_Get()
	if  s == DOTA_GAMERULES_STATE_PRE_GAME then
		SendToServerConsole( "dota_all_vision 1" )
		SendToServerConsole( "dota_creeps_no_spawning  1" )
		SendToServerConsole( "dota_dev forcegamestart" )

	elseif  s == DOTA_GAMERULES_STATE_GAME_IN_PROGRESS then
		GameRules:GetGameModeEntity():SetThink("Setup", self, 5)
	end
end

function CreepBlockAI:Setup()
	goodSpawn = Entities:FindByName( nil, "npc_dota_spawner_good_mid_staging" )
	goodWP = Entities:FindByName ( nil, "lane_mid_pathcorner_goodguys_1")
	heroSpawn = Entities:FindByName (nil, "dota_goodguys_tower2_mid")
	hero = Entities:FindByName (nil, "npc_dota_hero_nevermore")
	SendToServerConsole( "dota_create_item heart" )
	t1 =  Entities:FindByName(nil, "dota_goodguys_tower1_mid")
	t1Pos = t1:GetAbsOrigin()
	t1_c = t1Pos.y + t1Pos.x + 2000

	PlayerResource:SetCameraTarget(0, hero)

	heroSpeed = hero:GetBaseMoveSpeed()

	baseURL = "http://localhost:8080/creep_control"

	ai_state = STATE_GETMODEL
	ep = 1
	self:Reset()
	GameRules:GetGameModeEntity():SetThink("MainLoop", self)
	hero:SetContextThink("BotThink", function() return self:BotLoop() end, 0.2)
end

--------------------------------------------------------------------------------

STATE_GETMODEL = 0
STATE_SIMULATING = 1
STATE_SENDDATA = 2

function CreepBlockAI:MainLoop()
	if ai_state == STATE_GETMODEL then
		Say(hero, "Getting Latest Model..", false)
		request = CreateHTTPRequestScriptVM( "GET", baseURL .. "/get_model")
		request:Send( 	function( result )
							if result["StatusCode"] == 200 and ai_state == STATE_GETMODEL then
								local data = package.loaded['game/dkjson'].decode(result['Body'])
								self:UpdateModel(data)
								InitOU(self.mu, self.sigma)

								Say(hero, "Loaded Latest Model", false)
								Say(hero, "Starting Episode " .. data.next_ep, false)
								ep = data.next_ep -- get next episode from server

								self:Start()
								ai_state = STATE_SIMULATING
							end
						end )
	elseif ai_state == STATE_SIMULATING then

	elseif ai_state == STATE_SENDDATA then
		Say(hero, "Sending Experience..", false)
		request = CreateHTTPRequestScriptVM( "POST", baseURL .. "/update_model")
		request:SetHTTPRequestHeaderValue("Accept", "application/json")
		request:SetHTTPRequestRawPostBody('application/json', package.loaded['game/dkjson'].encode(SAR))
		request:Send( 	function( result )
							if result["StatusCode"] == 200 and ai_state == STATE_SENDDATA then
								Say(hero, "Model Updated", false)
								ai_state = STATE_GETMODEL
							end
						end )
	else
		Warning("Some shit has gone bad..")
	end

	return 3
end

function CreepBlockAI:BotLoop()
	if ai_state ~= STATE_SIMULATING then
		return 0.2
	end

	self:UpdatePositions() -- get hPos & cPos

	local terminal, action = self:UpdateSAR()
	if terminal then
		self:Reset()
		ResetOU()
		ai_state = STATE_SENDDATA
		return 0.2
	end

	hero:MoveToPosition(hPos + action)

	if t > 0 then
		Say(hero, "Gained Reward " .. SAR[t-1]['r'], false)
	end

	t = t + 1

	return 0.2
end

--------------------------------------------------------------------------------

function CreepBlockAI:UpdatePositions()
	hPos = hero:GetAbsOrigin()
	cPos = {}
	for i = 1,4 do
		cPos[i] = creeps[i]:GetAbsOrigin()
	end
end

function CreepBlockAI:UpdateSAR()
	local s_t = {}
	for i = 1,4 do
		s_t[i*2-1] = cPos[i].x - hPos.x
		s_t[i*2] = cPos[i].y - hPos.y
	end
	s_t[9] = heroSpeed
	s_t[10] = hPos.x
	s_t[11] = hPos.y

	local s_t_normal = {}
	for i = 1,11 do
		s_t_normal[i] = s_t[i] / 1000.0
	end
	local action = self:Run(s_t_normal)

	if t > 0 then
		local reward = 0
		for i = 1,4 do
			local dist = (cPos[i] - last_cPos[i]):Length2D() -- dif creep
			local hdist = (hPos - cPos[i]):Length2D() -- dif hero and creep
			if hdist < 500 then
				if dist < 20 then
					reward = reward + 0.5
				elseif dist < 40 then
					reward = reward + 0.35
				elseif dist < 60 then
					reward = reward + 0.2
				end
			end
		end

		SAR[t-1]['r'] = reward
		SAR[t-1]['s1'] = s_t
	end

	last_cPos = cPos

	local terminal = false
	local c1 = hPos.y + hPos.x + 100
	for i = 1,4 do
		local c2 = cPos[i].y - cPos[i].x
		local x = (c1 - c2) / 2.0
		local y = -x + c1
		if cPos[i].y > (-cPos[i].x + c1) then -- creep walk ahead hero
			SAR[t-1]['r'] = SAR[t-1]['r'] - 1
			SAR[t-1]['done'] = 1
			terminal = true
		end
		if cPos[i].y > (-cPos[i].x + t1_c) then
			SAR[t-1]['done'] = 1
			terminal = true
		end
	end

	if not terminal then
		SAR[t] = { s=s_t, a={action.x, action.y}, r=0, s1={}, done=0 }
	end

	return terminal, action
end

--------------------------------------------------------------------------------

function CreepBlockAI:Reset()
	hero:Stop()
	SendToServerConsole( "dota_dev hero_refresh" )
	FindClearSpaceForUnit(hero, heroSpawn:GetAbsOrigin() + Vector(150,-150,0), true)

	if creeps ~= nil then
		for i = 1,4 do
			creeps[i]:ForceKill(false)
		end
	end
end

function CreepBlockAI:Start()
	t = 0
	SAR = {}
	SAR['ep'] = ep

	creeps = {}
	for i=1,3 do
		creeps[i] = CreateUnitByName( "npc_dota_creep_goodguys_melee" , goodSpawn:GetAbsOrigin() + RandomVector( RandomFloat( 0, 200 ) ), true, nil, nil, DOTA_TEAM_GOODGUYS )
	end
	creeps[4] = CreateUnitByName( "npc_dota_creep_goodguys_ranged" , goodSpawn:GetAbsOrigin() + RandomVector( RandomFloat( 0, 200 ) ), true, nil, nil, DOTA_TEAM_GOODGUYS )

	for i = 1,4 do
		creeps[i]:SetInitialGoalEntity( goodWP )
	end
end

--------------------------------------------------------------------------------

function CreepBlockAI:UpdateModel(data)
	self.train = data.train_indicator
	self.explore = data.explore
	self.boot_strap = data.boot_strap
	self.ou = data.ou
	self.mu = data.mu
	self.sigma = data.sigma
	if ep == 1 or data.replace then
		self.W1 = data.W1
		self.b1 = data.b1
		self.W2 = data.W2
		self.b2 = data.b2
		self.W3 = data.W3
		self.b3 = data.b3
	end
end

function CreepBlockAI:Run(s_t)
	local action = Vector(0,0,0)
	local MAS = 100 -- Max Action Size

	if self.train == 1 and RandomInt(1,100) <= self.boot_strap then
		action = self:BestMoveEst()
		local action_size = math.sqrt(action.x * action.x + action.y + action.y)
		if action_size > MAS then
			-- scale down vector
			action.x = action.x * MAS / action_size
			action.y = action.y * MAS / action_size
		end
		DebugDrawCircle(hPos + action, Vector(0,255,0), 255, 25, true, 0.2)
	else
		local fc1 = RELU(FC(s_t, self.W1, self.b1))
		local fc2 = RELU(FC(fc1, self.W2, self.b2))
		local fc3 = TANH(FC(fc2, self.W3, self.b3))

		action = Vector(fc3[1]*100, fc3[2]*100, 0) -- scale up action to 10X (-10, 10)
		if self.train == 1 then
			if self.ou == 1 then
				local noise = CallOU()
				action.x = action.x + noise[1]
				action.y = action.y + noise[2]
			elseif self.explore ~= 0 then
				action.x = action.x + RandomFloat(-self.explore,self.explore)
				action.y = action.y + RandomFloat(-self.explore,self.explore)
			end
		end

		DebugDrawCircle(hPos + action, Vector(255,255,0), 255, 25, true, 0.2)
	end
	return action
end

-- Not used
function CreepBlockAI:ClosestCreep()
	local closest = 10000
	for i = 1,4 do
		local dist = (hPos - cPos[i]):Length2D()
		if dist < closest then
			closest = dist
		end
	end
	return closest
end

-- Not used
function CreepBlockAI:BestMoveEst()
	local action = Vector(0,0,0)

	local c1 = hPos.y + hPos.x
	local min_dist = 100000
	local target = Vector(0,0,0)
	for i = 1,4 do
		local c2 = cPos[i].y - cPos[i].x
		local x = (c1 - c2) / 2.0
		local y = -x + c1
		local dist = (Vector(x,y,0) - cPos[i]):Length2D()
		local creep_ahead = cPos[i].y > (-cPos[i].x + c1)

		if not creep_ahead and dist < min_dist then
			target = cPos[i] + Vector(100,100,0)
			min_dist = dist
		end
	end

	if target.x == 0 then
		action.x = 100
		action.y = 100
	else
		action.x = target.x - hPos.x
		action.y = target.y - hPos.y
	end

	return action
end

-- Neural Function

function FC(x, W, b)
	local y = {}
	for j = 1,#b do
		y[j] = 0
		for i = 1,#x do
			y[j] = y[j] + x[i]*W[i][j]
		end
		y[j] = y[j] + b[j]
	end
	return y
end

function RELU(x)
	local y = {}
	for i = 1,#x do
		if x[i] < 0 then
			y[i] = 0
		else
			y[i] = x[i]
		end
	end
	return y
end

function TANH(x)
	local y = {}
	for i = 1,#x do
		y[i] = (math.exp(x[i]) - math.exp(-x[i])) / (math.exp(x[i]) + math.exp(-x[i]))
	end
	return y
end

-- Ornstein Uhlenbeck
-- Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
function InitOU(mu, sigma)
	ou_mu = mu
	ou_sigma = sigma
	ou_theta = 0.15
	ou_dt = 1e-2
	x0 = nil
	ResetOU()
end

function ResetOU()
	if x0 == nil then
		x_prev1 = 0
		x_prev2 = 0
	else
		x_prev1 = x0
		x_prev2 = x0
	end
end

function CallOU()
	-- variance = std^2
	local average, variance = 0, 1
	local x1 = x_prev1 + ou_theta * (ou_mu - x_prev1) * ou_dt + ou_sigma * math.sqrt(ou_dt) * gaussian(average, variance)
	local x2 = x_prev2 + ou_theta * (ou_mu - x_prev2) * ou_dt + ou_sigma * math.sqrt(ou_dt) * gaussian(average, variance)
	x_prev1 = x1
	x_prev2 = x2
	return {x1, x2}
end

-- Normal Distribution Random
function gaussian (mean, variance)
    return  math.sqrt(-2 * variance * math.log(math.random())) *
            math.cos(2 * math.pi * math.random()) + mean
end

function mean (t)
    local sum = 0
    for k, v in pairs(t) do
        sum = sum + v
    end
    return sum / #t
end

function std (t)
    local squares, avg = 0, mean(t)
    for k, v in pairs(t) do
        squares = squares + ((avg - v) ^ 2)
    end
    local variance = squares / #t
    return math.sqrt(variance)
end

function showHistogram (t)
    local lo = math.ceil(math.min(unpack(t)))
    local hi = math.floor(math.max(unpack(t)))
    local hist, barScale = {}, 200
    for i = lo, hi do
        hist[i] = 0
        for k, v in pairs(t) do
            if math.ceil(v - 0.5) == i then
                hist[i] = hist[i] + 1
            end
        end
        io.write(i .. "\t" .. string.rep('=', hist[i] / #t * barScale))
        print(" " .. hist[i])
    end
end

dkjson = package.loaded['game/dkjson']

if CreepControl == nil then
	_G.CreepControl = class({})
end

-- Setup
HERO_NAME = "npc_dota_hero_sniper"

-- AI's state Definition
STATE_GETMODEL = 0
STATE_SIMULATING = 1
STATE_SENDDATA = 2

function Activate()
	GameRules.CreepControl = CreepControl()
	GameRules.CreepControl:InitGameMode()
end

function CreepControl:InitGameMode()
	print( "Template addon is loaded." )
	GameRules:SetShowcaseTime( 0 )
	GameRules:SetStrategyTime( 0 )
	GameRules:SetHeroSelectionTime( 0 )
	GameRules:GetGameModeEntity():SetCustomGameForceHero(HERO_NAME)

	ListenToGameEvent( "game_rules_state_change", Dynamic_Wrap( CreepControl, 'OnGameRulesStateChange' ), self )
end

function CreepControl:OnGameRulesStateChange()
	local s = GameRules:State_Get()
	if s == DOTA_GAMERULES_STATE_PRE_GAME then
		SendToServerConsole( "dota_all_vision 1" )
		SendToServerConsole( "dota_creeps_no_spawning  1" )
		SendToServerConsole( "dota_dev forcegamestart" )

	elseif s == DOTA_GAMERULES_STATE_GAME_IN_PROGRESS then
		GameRules:GetGameModeEntity():SetThink("Setup", self, 5)
	end
end

function CreepControl:Setup()
	goodSpawn = Entities:FindByName( nil, "npc_dota_spawner_good_mid_staging" )
	goodWP = Entities:FindByName ( nil, "lane_mid_pathcorner_goodguys_1")
	heroSpawn = Entities:FindByName (nil, "dota_goodguys_tower3_mid")
	hero = Entities:FindByName (nil, HERO_NAME)
	t1 = Entities:FindByName(nil, "dota_goodguys_tower1_mid")
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

function CreepControl:MainLoop()
	if ai_state == STATE_GETMODEL then
		Say(hero, "You are in STATE_GETMODEL", false)
		-- get model and update
		request = CreateHTTPRequestScriptVM( "GET", baseURL .. "/get_model")
		request:Send( 	function( result )
							if result["StatusCode"] == 200 and ai_state == STATE_GETMODEL then
								local data = dkjson.decode(result['Body'])
								-- self:UpdateModel(data)

								print(data)
								Say(hero, "Loaded Latest Model", false)
								Say(hero, "Starting Episode " .. ep, false)

								-- self:Start()
								ai_state = STATE_SIMULATING
							end
						end )

		-- start simulating
		-- ai_state = STATE_SIMULATING
	elseif ai_state == STATE_SIMULATING then
		Say(hero, "You are in STATE_SIMULATING", false)
	elseif ai_state == STATE_SENDDATA then
		Say(hero, "You are in STATE_SENDDATA", false)
		-- send data & get next episode
		ai_state = STATE_GETMODEL
	else
		Warning("Some thing went wrong!!!")
	end
	return 3
end

-- function CreepControl:Start()
-- 	t = 0
-- 	SAR = {}
-- 	SAR['ep'] = ep
-- 	ep = ep + 1
--
-- 	creeps = {}
-- 	for i = 1, 3 do
-- 		creeps[i] = CreateUnitByName( "npc_dota_creep_goodguys_melee", goodSpawn:GetAbsOrigin() + RandomVector( RandomFloat( 0, 200 ) ), true, nil, nil, DOTA_TEAM_GOODGUYS )
-- 	end
-- 	creeps[4] = CreateUnitByName( "npc_dota_creep_goodguys_ranged", goodSpawn:GetAbsOrigin() + RandomVector( RandomFloat( 0, 200 ) ), true, nil, nil, DOTA_TEAM_GOODGUYS )
--
-- 	for i = 1, 4 do
-- 		creeps[i]:SetInitialGoalEntity( goodWP )
-- 	end
-- end

function CreepControl:Reset()
	hero:Stop()
	SendToServerConsole( "dota_dev hero_refresh" )
	FindClearSpaceForUnit(hero, heroSpawn:GetAbsOrigin() + Vector(150, - 150, 0), true)

	-- if creeps ~= nil then
	-- 	for i = 1, 4 do
	-- 		creeps[i]:ForceKill(false)
	-- 	end
	-- end
end

function CreepControl:BotLoop()
	if ai_state ~= STATE_SIMULATING then
		return 0.2
	end

	self:UpdatePositions()
	--
	-- local terminal, action = self:UpdateSAR()
	-- if terminal then
	-- 	self:Reset()
	-- 	ai_state = STATE_SENDDATA
	-- 	return 0.2
	-- end
	--
	-- hero:MoveToPosition(hPos + action)
	--
	-- if t > 0 then
	-- 	Say(hero, "Gained Reward " .. SAR[t-1]['r'], false)
	-- end
	--
	-- t = t + 1

	local action = Vector(100, 100, 0)
	DebugDrawCircle(hero_pos + action, Vector(0,255,0), 255, 25, true, 0.2)
	hero:MoveToPosition(hero_pos + action)

	return 0.2
end

function CreepControl:UpdatePositions()
	hero_pos = hero:GetAbsOrigin()
	-- print(hero_pos)
end

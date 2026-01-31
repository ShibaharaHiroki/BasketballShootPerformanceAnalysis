/**
 * Sidebar component with parameter controls.
 */

import React, { useState, useEffect } from 'react';
import {
    Box,
    VStack,
    Heading,
    Button,
    Text,
    Slider,
    SliderTrack,
    SliderFilledTrack,
    SliderThumb,
    Divider,
    useToast,
    Menu,
    MenuButton,
    MenuList,
    MenuItem,
    Collapse,
    IconButton,
    HStack,
} from '@chakra-ui/react';
import { ChevronDownIcon, ChevronUpIcon, SettingsIcon } from '@chakra-ui/icons';
import { useAppContext } from '../context/AppContext';
import { apiClient } from '../services/api';
import type { ClassWeight, AvailablePlayer } from '../types';

// ============================================
// CONFIGURATION: Change this to switch leagues
// Options: 'nba' | 'bleague'
// ============================================
const SELECTED_LEAGUE = 'nba' as 'nba' | 'bleague';

const PLAYER_IDS_NBA = [203999, 203507, 203954];
const PLAYER_NAMES_NBA = ['Jokic', 'Antetokounmpo', 'Embiid'];

const Sidebar: React.FC = () => {
    const {
        cluster1,
        cluster2,
        resetClusters,
        tensorShape,
        playerNames,
        setEmbedding,
        setScaledData,
        setProjMats,
        setIsLoading,
        setPlayerNames,
        setPlayerLabels,
        setGameIds,
        setMetadata,
        setTensorShape,
        setCurrentLeague,
    } = useAppContext();

    const toast = useToast();

    // League selection state (can be switched in hidden UI)
    const [selectedLeague, setSelectedLeagueLocal] = useState<'nba' | 'bleague'>(SELECTED_LEAGUE);

    // Analysis mode state (player analysis vs team season comparison)
    const [analysisMode, setAnalysisMode] = useState<'player' | 'team_season'>('player');

    // Update context when league changes
    const setSelectedLeague = (league: 'nba' | 'bleague') => {
        setSelectedLeagueLocal(league);
        setCurrentLeague(league);
        // Reset analysis mode to player when switching leagues
        setAnalysisMode('player');
    };

    // Player selection state
    const [availablePlayers, setAvailablePlayers] = useState<AvailablePlayer[]>([]);
    const [selectedPlayerIds, setSelectedPlayerIds] = useState<number[]>([]);
    const [isFetchingPlayers, setIsFetchingPlayers] = useState(false);

    // Auto-fetch players when league changes
    useEffect(() => {
        const fetchPlayers = async () => {
            setIsFetchingPlayers(true);
            setAvailablePlayers([]);
            setSelectedPlayerIds([]);
            try {
                const response = await apiClient.getPlayers([2022], selectedLeague);
                setAvailablePlayers(response.players);
                // Pre-select first 3 players
                const defaultSelection = response.players.slice(0, 3).map(p => p.player_id);
                setSelectedPlayerIds(defaultSelection);
            } catch (err) {
                console.error(`Failed to fetch ${selectedLeague} players:`, err);
            } finally {
                setIsFetchingPlayers(false);
            }
        };
        fetchPlayers();
    }, [selectedLeague]);

    // Current player names for weight tuning (use from context if available, fallback to defaults)
    const currentPlayerNames = playerNames.length > 0 ? playerNames : PLAYER_NAMES_NBA;

    // TULCA dimensions
    const [sDim, setSDim] = useState(4);
    const [vDim, setVDim] = useState(150);
    const [tulcaChannel, setTulcaChannel] = useState(0);  // 0=attempts, 1=makes, 2=weighted

    // Class weights
    const [selectedClass, setSelectedClass] = useState(0);
    const [classWeights, setClassWeights] = useState<ClassWeight[]>([
        { w_tg: 0.0, w_bw: 1.0, w_bg: 1.0 },
        { w_tg: 0.0, w_bw: 1.0, w_bg: 1.0 },
        { w_tg: 0.0, w_bw: 1.0, w_bg: 1.0 },
    ]);

    // Advanced settings visibility (hidden by default)
    const [showAdvanced, setShowAdvanced] = useState(false);
    // Whether advanced section is revealed (hidden by default, double-click to reveal)
    const [advancedRevealed, setAdvancedRevealed] = useState(false);

    // Ensure selectedClass is within bounds
    const safeSelectedClass = Math.min(selectedClass, classWeights.length - 1);
    const currentWeight = classWeights[safeSelectedClass] || { w_tg: 1.0, w_bw: 1.0, w_bg: 1.0 };

    // Note: League is fixed to B.League, no handleLeagueChange needed

    const handleFetchPlayers = async () => {
        setIsFetchingPlayers(true);
        try {
            const response = await apiClient.getPlayers([2022], selectedLeague);
            setAvailablePlayers(response.players);
            toast({
                title: 'Players Loaded',
                description: `Found ${response.players.length} players`,
                status: 'success',
                duration: 2000,
                isClosable: true,
            });
        } catch (err: any) {
            toast({
                title: 'Error',
                description: err.response?.data?.detail || 'Failed to fetch players',
                status: 'error',
                duration: 3000,
                isClosable: true,
            });
        } finally {
            setIsFetchingPlayers(false);
        }
    };

    const handleTogglePlayer = (playerId: number) => {
        setSelectedPlayerIds((prev) =>
            prev.includes(playerId)
                ? prev.filter((id) => id !== playerId)
                : [...prev, playerId]
        );
    };

    const handleApplyPlayerSelection = async () => {
        // Team season mode doesn't need player selection
        if (analysisMode === 'team_season') {
            setIsLoading(true);
            try {
                const response = await apiClient.initialize(
                    [],  // No player IDs needed for team season
                    [2022],
                    sDim,
                    vDim,
                    tulcaChannel,
                    selectedLeague,
                    analysisMode
                );
                setEmbedding(response.embedding);
                setScaledData(response.scaled_data);
                setProjMats(response.proj_mats);
                setPlayerLabels(response.player_labels);
                setGameIds(response.game_ids);
                setPlayerNames(response.player_names);
                setTensorShape(response.tensor_shape);
                setMetadata(response.metadata);

                // Update class weights for 2 seasons
                setClassWeights([
                    { w_tg: 0.0, w_bw: 1.0, w_bg: 1.0 },
                    { w_tg: 0.0, w_bw: 1.0, w_bg: 1.0 },
                ]);
                setSelectedClass(0);

                toast({
                    title: 'Team Season Comparison',
                    description: 'Comparing 2022-23 vs 2023-24 seasons',
                    status: 'success',
                    duration: 2000,
                    isClosable: true,
                });
            } catch (err: any) {
                toast({
                    title: 'Error',
                    description: err.response?.data?.detail || 'Failed to initialize team season comparison',
                    status: 'error',
                    duration: 3000,
                    isClosable: true,
                });
            } finally {
                setIsLoading(false);
            }
            return;
        }

        // Player analysis mode
        if (selectedPlayerIds.length < 2) {
            toast({
                title: 'Invalid Selection',
                description: 'Please select at least 2 players',
                status: 'warning',
                duration: 3000,
                isClosable: true,
            });
            return;
        }

        setIsLoading(true);
        try {
            const response = await apiClient.initialize(
                selectedPlayerIds,
                [2022],
                sDim,
                vDim,
                tulcaChannel,
                selectedLeague,
                analysisMode
            );
            setEmbedding(response.embedding);
            setScaledData(response.scaled_data);
            setProjMats(response.proj_mats);
            setPlayerLabels(response.player_labels);
            setGameIds(response.game_ids);
            setPlayerNames(response.player_names);
            setTensorShape(response.tensor_shape);
            setMetadata(response.metadata);

            // Update class weights array to match number of players
            const newWeights = selectedPlayerIds.map(() => ({ w_tg: 0.0, w_bw: 1.0, w_bg: 1.0 }));
            setClassWeights(newWeights);
            setSelectedClass(0);

            toast({
                title: 'Players Applied',
                description: `Initialized with ${selectedPlayerIds.length} players`,
                status: 'success',
                duration: 2000,
                isClosable: true,
            });
        } catch (err: any) {
            toast({
                title: 'Error',
                description: err.response?.data?.detail || 'Failed to initialize with selected players',
                status: 'error',
                duration: 3000,
                isClosable: true,
            });
        } finally {
            setIsLoading(false);
        }
    };

    const handleWeightChange = (key: 'w_tg' | 'w_bw' | 'w_bg', value: number) => {
        const newWeights = [...classWeights];
        const targetIndex = Math.min(selectedClass, newWeights.length - 1);
        newWeights[targetIndex] = { ...newWeights[targetIndex], [key]: value };
        setClassWeights(newWeights);
    };

    const handleApplyWeights = async () => {
        setIsLoading(true);
        try {
            const response = await apiClient.recomputeTulca(classWeights, sDim, vDim, tulcaChannel);
            setEmbedding(response.embedding);
            setScaledData(response.scaled_data);
            setProjMats(response.proj_mats);

            toast({
                title: 'TULCA Recomputed',
                description: 'Embedding updated successfully',
                status: 'success',
                duration: 2000,
                isClosable: true,
            });
        } catch (err: any) {
            toast({
                title: 'Error',
                description: err.response?.data?.detail || 'Failed to recompute TULCA',
                status: 'error',
                duration: 3000,
                isClosable: true,
            });
        } finally {
            setIsLoading(false);
        }
    };

    const [S, V] = tensorShape.length >= 3 ? [tensorShape[1], tensorShape[2]] : [4, 272];

    return (
        <Box p={4} h="100vh" overflowY="auto" position="relative">
            <VStack spacing={4} align="stretch">

                {/* Metric Selection - Always visible */}
                <Box>
                    <Text fontWeight="bold" fontSize="sm" mb={2} color="white">
                        Metric
                    </Text>
                    <Menu>
                        <MenuButton
                            as={Button}
                            size="sm"
                            rightIcon={<ChevronDownIcon />}
                            bg="gray.800"
                            color="white"
                            _hover={{ bg: 'gray.700' }}
                            _active={{ bg: 'gray.600' }}
                            w="100%"
                            textAlign="left"
                            fontWeight="normal"
                            mb={3}
                        >
                            {tulcaChannel === 0 ? 'Attempts' : tulcaChannel === 1 ? 'Makes' : tulcaChannel === 2 ? 'Points' : 'Misses'}
                        </MenuButton>
                        <MenuList bg="gray.800" borderColor="gray.600">
                            <MenuItem bg="gray.800" color="white" _hover={{ bg: 'gray.700' }} onClick={() => setTulcaChannel(0)}>Attempts</MenuItem>
                            <MenuItem bg="gray.800" color="white" _hover={{ bg: 'gray.700' }} onClick={() => setTulcaChannel(1)}>Makes</MenuItem>
                            <MenuItem bg="gray.800" color="white" _hover={{ bg: 'gray.700' }} onClick={() => setTulcaChannel(2)}>Points</MenuItem>
                            <MenuItem bg="gray.800" color="white" _hover={{ bg: 'gray.700' }} onClick={() => setTulcaChannel(4)}>Misses</MenuItem>
                        </MenuList>
                    </Menu>
                </Box>

                <Divider borderColor="gray.700" />

                {/* Advanced Settings Toggle - Only show if revealed */}
                {advancedRevealed && (
                    <HStack
                        justify="space-between"
                        cursor="pointer"
                        onClick={() => setShowAdvanced(!showAdvanced)}
                        _hover={{ bg: 'gray.800' }}
                        p={2}
                        borderRadius="md"
                    >
                        <HStack>
                            <SettingsIcon color="gray.400" />
                            <Text fontWeight="bold" fontSize="sm" color="gray.400">
                                Advanced Settings
                            </Text>
                        </HStack>
                        {showAdvanced ? <ChevronUpIcon color="gray.400" /> : <ChevronDownIcon color="gray.400" />}
                    </HStack>
                )}

                {/* Collapsible Advanced Settings */}
                <Collapse in={showAdvanced} animateOpacity>
                    <VStack spacing={4} align="stretch" pl={2} borderLeftWidth="2px" borderColor="gray.700">

                        {/* League Selection */}
                        <Box>
                            <Text fontWeight="bold" fontSize="xs" mb={2} color="gray.400">
                                League
                            </Text>
                            <Menu>
                                <MenuButton
                                    as={Button}
                                    size="xs"
                                    rightIcon={<ChevronDownIcon />}
                                    bg="gray.800"
                                    color="white"
                                    _hover={{ bg: 'gray.700' }}
                                    _active={{ bg: 'gray.600' }}
                                    w="100%"
                                    textAlign="left"
                                    fontWeight="normal"
                                >
                                    {selectedLeague === 'nba' ? 'NBA' : 'B.League'}
                                </MenuButton>
                                <MenuList bg="gray.800" borderColor="gray.600">
                                    <MenuItem
                                        bg={selectedLeague === 'nba' ? 'gray.700' : 'gray.800'}
                                        color="white"
                                        _hover={{ bg: 'gray.700' }}
                                        fontSize="xs"
                                        onClick={() => setSelectedLeague('nba')}
                                    >
                                        NBA
                                    </MenuItem>
                                    <MenuItem
                                        bg={selectedLeague === 'bleague' ? 'gray.700' : 'gray.800'}
                                        color="white"
                                        _hover={{ bg: 'gray.700' }}
                                        fontSize="xs"
                                        onClick={() => setSelectedLeague('bleague')}
                                    >
                                        B.League
                                    </MenuItem>
                                </MenuList>
                            </Menu>
                        </Box>

                        {/* Analysis Mode Selection (B.League only) */}
                        {selectedLeague === 'bleague' && (
                            <Box>
                                <Text fontWeight="bold" fontSize="xs" mb={2} color="gray.400">
                                    Analysis Mode
                                </Text>
                                <Menu>
                                    <MenuButton
                                        as={Button}
                                        size="xs"
                                        rightIcon={<ChevronDownIcon />}
                                        bg="gray.800"
                                        color="white"
                                        _hover={{ bg: 'gray.700' }}
                                        _active={{ bg: 'gray.600' }}
                                        w="100%"
                                        textAlign="left"
                                        fontWeight="normal"
                                    >
                                        {analysisMode === 'player' ? 'Player Analysis' : 'Team Season Comparison'}
                                    </MenuButton>
                                    <MenuList bg="gray.800" borderColor="gray.600">
                                        <MenuItem
                                            bg={analysisMode === 'player' ? 'gray.700' : 'gray.800'}
                                            color="white"
                                            _hover={{ bg: 'gray.700' }}
                                            fontSize="xs"
                                            onClick={() => setAnalysisMode('player')}
                                        >
                                            Player Analysis
                                        </MenuItem>
                                        <MenuItem
                                            bg={analysisMode === 'team_season' ? 'gray.700' : 'gray.800'}
                                            color="white"
                                            _hover={{ bg: 'gray.700' }}
                                            fontSize="xs"
                                            onClick={() => setAnalysisMode('team_season')}
                                        >
                                            Team Season Comparison (三遠)
                                        </MenuItem>
                                    </MenuList>
                                </Menu>
                            </Box>
                        )}

                        {/* Player Selection */}
                        <Box>
                            <Text fontWeight="bold" fontSize="xs" mb={2} color="gray.400">
                                Player Selection
                            </Text>

                            <Button
                                colorScheme="purple"
                                size="xs"
                                w="100%"
                                mb={2}
                                onClick={handleFetchPlayers}
                                isLoading={isFetchingPlayers}
                            >
                                Fetch Players
                            </Button>

                            {availablePlayers.length > 0 && (
                                <>
                                    <VStack align="stretch" maxH="150px" overflowY="auto" spacing={1} mb={2} p={2} borderWidth="1px" borderRadius="md" borderColor="gray.700" bg="gray.900">
                                        {availablePlayers.map((player) => (
                                            <Box
                                                key={player.player_id}
                                                display="flex"
                                                alignItems="center"
                                                fontSize="10px"
                                                cursor="pointer"
                                                onClick={() => handleTogglePlayer(player.player_id)}
                                                _hover={{ bg: 'gray.800' }}
                                                p={1}
                                                borderRadius="sm"
                                            >
                                                <input
                                                    type="checkbox"
                                                    checked={selectedPlayerIds.includes(player.player_id)}
                                                    onChange={() => handleTogglePlayer(player.player_id)}
                                                    style={{ marginRight: '6px' }}
                                                />
                                                <Text flex="1" color="white">{player.player_name}</Text>
                                                <Text color="gray.500" fontSize="9px">({player.game_count}g)</Text>
                                            </Box>
                                        ))}
                                    </VStack>

                                    <Text fontSize="10px" mb={2} color="gray.400">
                                        Selected: {selectedPlayerIds.length}
                                    </Text>

                                    <Button
                                        colorScheme="green"
                                        size="xs"
                                        w="100%"
                                        onClick={handleApplyPlayerSelection}
                                        isDisabled={selectedPlayerIds.length < 2}
                                    >
                                        Apply
                                    </Button>
                                </>
                            )}
                        </Box>

                        <Divider borderColor="gray.700" />

                        {/* TULCA Dimensions */}
                        <Box>
                            <Text fontWeight="bold" fontSize="xs" mb={2} color="gray.400">
                                Dimensions
                            </Text>

                            <Text fontSize="10px" mb={1} color="gray.400">
                                time: {sDim}
                            </Text>
                            <Slider
                                value={sDim}
                                onChange={setSDim}
                                min={1}
                                max={S}
                                step={1}
                                mb={2}
                                size="sm"
                            >
                                <SliderTrack>
                                    <SliderFilledTrack />
                                </SliderTrack>
                                <SliderThumb />
                            </Slider>

                            <Text fontSize="10px" mb={1} color="gray.400">
                                space: {vDim}
                            </Text>
                            <Slider
                                value={vDim}
                                onChange={setVDim}
                                min={1}
                                max={V}
                                step={1}
                                mb={2}
                                size="sm"
                            >
                                <SliderTrack>
                                    <SliderFilledTrack />
                                </SliderTrack>
                                <SliderThumb />
                            </Slider>
                        </Box>
                    </VStack>
                </Collapse>

                <Divider />

                {/* Weight Tuning */}
                <Box>
                    <Text fontWeight="bold" fontSize="sm" mb={2} color="white">
                        Weights
                    </Text>

                    <Text fontSize="xs" mb={1} color="white">
                        Player:
                    </Text>
                    <Menu>
                        <MenuButton
                            as={Button}
                            size="sm"
                            rightIcon={<ChevronDownIcon />}
                            bg="gray.800"
                            color="white"
                            _hover={{ bg: 'gray.700' }}
                            _active={{ bg: 'gray.600' }}
                            w="100%"
                            textAlign="left"
                            fontWeight="normal"
                            mb={3}
                        >
                            {currentPlayerNames[selectedClass] || 'Select Player'}
                        </MenuButton>
                        <MenuList bg="gray.800" borderColor="gray.600">
                            {currentPlayerNames.map((name, idx) => (
                                <MenuItem
                                    key={idx}
                                    bg="gray.800"
                                    color="white"
                                    _hover={{ bg: 'gray.700' }}
                                    onClick={() => setSelectedClass(idx)}
                                >
                                    {name}
                                </MenuItem>
                            ))}
                        </MenuList>
                    </Menu>

                    <Text fontSize="xs" mb={1} color="white">
                        target: {currentWeight.w_tg.toFixed(1)}
                    </Text>
                    <Slider
                        value={currentWeight.w_tg}
                        onChange={(val) => handleWeightChange('w_tg', val)}
                        min={0}
                        max={1}
                        step={0.1}
                        mb={3}
                    >
                        <SliderTrack>
                            <SliderFilledTrack />
                        </SliderTrack>
                        <SliderThumb />
                    </Slider>

                    <Text fontSize="xs" mb={1} color="white">
                        difference: {currentWeight.w_bw.toFixed(1)}
                    </Text>
                    <Slider
                        value={currentWeight.w_bw}
                        onChange={(val) => handleWeightChange('w_bw', val)}
                        min={0}
                        max={1}
                        step={0.1}
                        mb={3}
                    >
                        <SliderTrack>
                            <SliderFilledTrack />
                        </SliderTrack>
                        <SliderThumb />
                    </Slider>

                    <Text fontSize="xs" mb={1} color="white">
                        non_target: {currentWeight.w_bg.toFixed(1)}
                    </Text>
                    <Slider
                        value={currentWeight.w_bg}
                        onChange={(val) => handleWeightChange('w_bg', val)}
                        min={0}
                        max={1}
                        step={0.1}
                        mb={3}
                    >
                        <SliderTrack>
                            <SliderFilledTrack />
                        </SliderTrack>
                        <SliderThumb />
                    </Slider>

                    <Button colorScheme="blue" size="sm" w="100%" onClick={handleApplyWeights}>
                        Apply
                    </Button>

                    <Button colorScheme="gray" size="sm" w="100%" onClick={resetClusters} mt={2}>
                        Reset Selection
                    </Button>
                </Box>

                {/* Small icon to reveal Advanced Settings */}
                <Box
                    position="absolute"
                    bottom={2}
                    left={2}
                    cursor="pointer"
                    onClick={() => setAdvancedRevealed(!advancedRevealed)}
                    opacity={0.3}
                    _hover={{ opacity: 0.7 }}
                    transition="opacity 0.2s"
                >
                    <SettingsIcon boxSize={3} color="gray.500" />
                </Box>
            </VStack>
        </Box>
    );
};

export default Sidebar;

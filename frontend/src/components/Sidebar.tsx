/**
 * Sidebar component with parameter controls.
 */

import React, { useState } from 'react';
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
    Select,
    Divider,
    useToast,
} from '@chakra-ui/react';
import { useAppContext } from '../context/AppContext';
import { apiClient } from '../services/api';
import type { ClassWeight, AvailablePlayer } from '../types';

const PLAYER_IDS = [203999, 203507, 203954];
const PLAYER_NAMES = ['Jokic', 'Antetokounmpo', 'Embiid'];

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
    } = useAppContext();

    const toast = useToast();

    // Player selection state
    const [availablePlayers, setAvailablePlayers] = useState<AvailablePlayer[]>([]);
    const [selectedPlayerIds, setSelectedPlayerIds] = useState<number[]>(PLAYER_IDS);
    const [isFetchingPlayers, setIsFetchingPlayers] = useState(false);

    // Current player names for weight tuning (use from context if available, fallback to defaults)
    const currentPlayerNames = playerNames.length > 0 ? playerNames : PLAYER_NAMES;

    // TULCA dimensions
    const [sDim, setSDim] = useState(4);
    const [vDim, setVDim] = useState(135);
    const [tulcaChannel, setTulcaChannel] = useState(0);  // 0=attempts, 1=makes, 2=weighted

    // Class weights
    const [selectedClass, setSelectedClass] = useState(0);
    const [classWeights, setClassWeights] = useState<ClassWeight[]>([
        { w_tg: 0.0, w_bw: 1.0, w_bg: 1.0 },
        { w_tg: 0.0, w_bw: 1.0, w_bg: 1.0 },
        { w_tg: 0.0, w_bw: 1.0, w_bg: 1.0 },
    ]);

    // Ensure selectedClass is within bounds
    const safeSelectedClass = Math.min(selectedClass, classWeights.length - 1);
    const currentWeight = classWeights[safeSelectedClass] || { w_tg: 1.0, w_bw: 1.0, w_bg: 1.0 };

    const handleFetchPlayers = async () => {
        setIsFetchingPlayers(true);
        try {
            const response = await apiClient.getPlayers([2022]);
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
            const response = await apiClient.initialize(selectedPlayerIds, [2022], sDim, vDim, tulcaChannel);
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
        <Box p={4} h="100vh" overflowY="auto">
            <VStack spacing={4} align="stretch">
                <Divider borderColor="gray.700" />


                {/* Player Selection - HIDDEN */}
                {/* 
                <Box>
                    <Text fontWeight="bold" fontSize="sm" mb={2} color="white">
                        Player Selection
                    </Text>

                    <Button
                        colorScheme="purple"
                        size="sm"
                        w="100%"
                        mb={3}
                        onClick={handleFetchPlayers}
                        isLoading={isFetchingPlayers}
                    >
                        Fetch Available Players
                    </Button>

                    {availablePlayers.length > 0 && (
                        <>
                            <VStack align="stretch" maxH="200px" overflowY="auto" spacing={1} mb={3} p={2} borderWidth="1px" borderRadius="md" borderColor="gray.700" bg="gray.900">
                                {availablePlayers.map((player) => (
                                    <Box
                                        key={player.player_id}
                                        display="flex"
                                        alignItems="center"
                                        fontSize="xs"
                                        cursor="pointer"
                                        onClick={() => handleTogglePlayer(player.player_id)}
                                        _hover={{ bg: 'gray.100' }}
                                        p={1}
                                        borderRadius="sm"
                                    >
                                        <input
                                            type="checkbox"
                                            checked={selectedPlayerIds.includes(player.player_id)}
                                            onChange={() => handleTogglePlayer(player.player_id)}
                                            style={{ marginRight: '8px' }}
                                        />
                                        <Text flex="1">{player.player_name}</Text>
                                        <Text color="white" fontSize="10px">({player.game_count} games)</Text>
                                    </Box>
                                ))}
                            </VStack>

                            <Text fontSize="xs" mb={2} color="white">
                                Selected: {selectedPlayerIds.length} players
                            </Text>

                            <Button
                                colorScheme="green"
                                size="sm"
                                w="100%"
                                onClick={handleApplyPlayerSelection}
                                isDisabled={selectedPlayerIds.length < 2}
                            >
                                Apply Player Selection
                            </Button>
                        </>
                    )}
                </Box>
                */}

                {/* Current Selected Players Display - HIDDEN */}
                {/* 
                {currentPlayerNames.length > 0 && currentPlayerNames !== PLAYER_NAMES && (
                    <Box>
                        <Text fontWeight="bold" fontSize="sm" mb={2} color="white">
                            Current Players ({currentPlayerNames.length}):
                        </Text>
                        <VStack align="stretch" spacing={1} fontSize="xs" p={2} borderWidth="1px" borderRadius="md" borderColor="gray.700" bg="gray.900">
                            {currentPlayerNames.map((name, idx) => (
                                <Text key={idx} color="white">
                                    {idx + 1}. {name}
                                </Text>
                            ))}
                        </VStack>
                    </Box>
                )}
                */}

                <Divider />

                {/* Metric Selection */}
                <Box>
                    <Text fontWeight="bold" fontSize="sm" mb={2} color="white">
                        Metric
                    </Text>
                    <Select
                        size="sm"
                        value={tulcaChannel}
                        onChange={(e) => setTulcaChannel(Number(e.target.value))}
                        mb={3}
                        color="white"
                        bg="gray.800"
                    >
                        <option value={0} style={{ color: 'black' }}>Attempts</option>
                        <option value={1} style={{ color: 'black' }}>Makes</option>
                        <option value={2} style={{ color: 'black' }}>Points</option>
                        <option value={3} style={{ color: 'black' }}>Misses</option>
                    </Select>
                </Box>

                <Divider />

                {/* TULCA Dimensions */}
                <Box>
                    <Text fontWeight="bold" fontSize="sm" mb={2} color="white">
                        Dimensions
                    </Text>

                    <Text fontSize="xs" mb={1} color="white">
                        time: {sDim}
                    </Text>
                    <Slider
                        value={sDim}
                        onChange={setSDim}
                        min={1}
                        max={S}
                        step={1}
                        mb={3}
                    >
                        <SliderTrack>
                            <SliderFilledTrack />
                        </SliderTrack>
                        <SliderThumb />
                    </Slider>

                    <Text fontSize="xs" mb={1} color="white">
                        space: {vDim}
                    </Text>
                    <Slider
                        value={vDim}
                        onChange={setVDim}
                        min={1}
                        max={V}
                        step={1}
                        mb={3}
                    >
                        <SliderTrack>
                            <SliderFilledTrack />
                        </SliderTrack>
                        <SliderThumb />
                    </Slider>
                </Box>

                <Divider />

                {/* Weight Tuning */}
                <Box>
                    <Text fontWeight="bold" fontSize="sm" mb={2} color="white">
                        Weights
                    </Text>

                    <Text fontSize="xs" mb={1} color="white">
                        Player:
                    </Text>
                    <Select
                        size="sm"
                        value={selectedClass}
                        onChange={(e) => setSelectedClass(Number(e.target.value))}
                        mb={3}
                        color="white"
                        bg="gray.800"
                    >
                        {currentPlayerNames.map((name, idx) => (
                            <option key={idx} value={idx} style={{ color: 'black' }}>
                                {name}
                            </option>
                        ))}
                    </Select>

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
            </VStack>
        </Box>
    );
};

export default Sidebar;

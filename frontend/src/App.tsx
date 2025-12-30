/**
 * Main App component with Chakra UI and layout.
 */

import React, { useEffect } from 'react';
import { ChakraProvider, Box, Flex, useToast, extendTheme } from '@chakra-ui/react';
import { AppProvider, useAppContext } from './context/AppContext';
import Sidebar from './components/Sidebar';
import ScatterPlot from './components/ScatterPlot';
import SpatialHeatmap from './components/SpatialHeatmap';
import RawDataExplorer from './components/RawDataExplorer';
import { apiClient } from './services/api';

// Custom theme with responsive font sizes
const theme = extendTheme({
    styles: {
        global: {
            'html': {
                fontSize: 'clamp(10px, 0.8vw, 16px)', // Scales from 10px to 16px based on viewport
            },
        },
    },
});

const AppContent: React.FC = () => {
    const {
        setEmbedding,
        setScaledData,
        setProjMats,
        setPlayerLabels,
        setGameIds,
        setPlayerNames,
        setTensorShape,
        setMetadata,
        setIsLoading,
        setError,
        isLoading,
        error,
    } = useAppContext();

    const toast = useToast();

    useEffect(() => {
        // Initialize data on mount
        const initializeData = async () => {
            setIsLoading(true);
            setError(null);

            // Show info toast about long loading time
            toast({
                title: 'Loading NBA Data',
                description: 'Downloading and processing NBA shot data. This may take 2-3 minutes...',
                status: 'info',
                duration: 10000,
                isClosable: true,
            });

            try {
                const data = await apiClient.initialize();
                setEmbedding(data.embedding);
                setScaledData(data.scaled_data);
                setProjMats(data.proj_mats);
                setPlayerLabels(data.player_labels);
                setGameIds(data.game_ids);
                setPlayerNames(data.player_names);
                setTensorShape(data.tensor_shape);
                setMetadata(data.metadata);

                toast({
                    title: 'Data Loaded',
                    description: `Loaded ${data.game_ids.length} games successfully`,
                    status: 'success',
                    duration: 3000,
                    isClosable: true,
                });
            } catch (err: any) {
                console.error('Full error object:', err);
                console.error('Error response:', err.response);
                console.error('Error request:', err.request);
                console.error('Error message:', err.message);
                console.error('Error config:', err.config);

                const errorMsg = err.response?.data?.detail || err.message || 'Failed to load data';
                setError(errorMsg);
                toast({
                    title: 'Error',
                    description: `${errorMsg}. Check console for details.`,
                    status: 'error',
                    duration: 10000,
                    isClosable: true,
                });
            } finally {
                setIsLoading(false);
            }
        };

        initializeData();
    }, []);

    if (error && isLoading === false) {
        return (
            <Box p={8} color="red.500">
                Error: {error}
            </Box>
        );
    }

    return (
        <Flex h="100vh" w="100vw" overflow="hidden">
            {/* Sidebar */}
            <Box flex="1" borderRight="1px" borderColor="gray.700" bg="black" overflowY="auto">
                <Sidebar />
            </Box>

            {/* Middle Column: Scatter Plot + Spatial Heatmap */}
            <Box flex="4" p={2} display="flex" flexDirection="column" gap={2} bg="gray.900">
                <Box flex="1" minH="0" overflow="hidden">
                    <ScatterPlot />
                </Box>
                <Box flex="1" minH="0" overflow="hidden">
                    <SpatialHeatmap />
                </Box>
            </Box>

            {/* Right Panel: Raw Data Explorer */}
            <Box flex="5" borderLeft="1px" borderColor="gray.700" overflow="hidden">
                <RawDataExplorer />
            </Box>
        </Flex>
    );
};

const App: React.FC = () => {
    return (
        <ChakraProvider theme={theme}>
            <AppProvider>
                <AppContent />
            </AppProvider>
        </ChakraProvider>
    );
};

export default App;

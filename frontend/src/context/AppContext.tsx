/**
 * React Context for global application state.
 */

import React, { createContext, useContext, useState, ReactNode } from 'react';
import type { AppState } from '../types';

interface AppContextType extends AppState {
    setEmbedding: (embedding: number[][]) => void;
    setScaledData: (data: number[][]) => void;
    setProjMats: (mats: number[][][]) => void;
    setPlayerLabels: (labels: number[]) => void;
    setGameIds: (ids: number[]) => void;
    setPlayerNames: (names: string[]) => void;
    setTensorShape: (shape: number[]) => void;
    setMetadata: (metadata: any) => void;
    setCluster1: (indices: number[] | null) => void;
    setCluster2: (indices: number[] | null) => void;
    setContribTensor: (tensor: number[][] | null) => void;
    setIsLoading: (loading: boolean) => void;
    setError: (error: string | null) => void;
    resetClusters: () => void;
    setTimeBin: (bin: string) => void;
    setClusterFilter: (filter: 'both' | 'c1' | 'c2') => void;
    setContribData: (data: number[][] | null) => void;
    setDomData: (data: number[][] | null) => void;
    currentLeague: 'nba' | 'bleague';
    setCurrentLeague: (league: 'nba' | 'bleague') => void;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

export const AppProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
    const [embedding, setEmbedding] = useState<number[][]>([]);
    const [scaledData, setScaledData] = useState<number[][]>([]);
    const [projMats, setProjMats] = useState<number[][][]>([]);
    const [playerLabels, setPlayerLabels] = useState<number[]>([]);
    const [gameIds, setGameIds] = useState<number[]>([]);
    const [playerNames, setPlayerNames] = useState<string[]>([]);
    const [tensorShape, setTensorShape] = useState<number[]>([]);
    const [metadata, setMetadata] = useState<any>({});
    const [cluster1, setCluster1] = useState<number[] | null>(null);
    const [cluster2, setCluster2] = useState<number[] | null>(null);
    const [contribTensor, setContribTensor] = useState<number[][] | null>(null);
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);
    const [currentLeague, setCurrentLeague] = useState<'nba' | 'bleague'>('nba');
    const [timeBin, setTimeBin] = useState<string>('all');
    const [clusterFilter, setClusterFilter] = useState<'both' | 'c1' | 'c2'>('both');
    const [contribData, setContribData] = useState<number[][] | null>(null);
    const [domData, setDomData] = useState<number[][] | null>(null);

    const resetClusters = () => {
        setCluster1(null);
        setCluster2(null);
        setContribTensor(null);
        setContribData(null);
        setDomData(null);
        setTimeBin('all');
        setClusterFilter('both');
    };

    const value: AppContextType = {
        embedding,
        scaledData,
        projMats,
        playerLabels,
        gameIds,
        playerNames,
        tensorShape,
        metadata,
        cluster1,
        cluster2,
        contribTensor,
        isLoading,
        error,
        timeBin,
        clusterFilter,
        contribData,
        domData,
        setEmbedding,
        setScaledData,
        setProjMats,
        setPlayerLabels,
        setGameIds,
        setPlayerNames,
        setTensorShape,
        setMetadata,
        setCluster1,
        setCluster2,
        setContribTensor,
        setIsLoading,
        setError,
        resetClusters,
        currentLeague,
        setCurrentLeague,
        setTimeBin,
        setClusterFilter,
        setContribData,
        setDomData,
    };

    return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
};

export const useAppContext = () => {
    const context = useContext(AppContext);
    if (context === undefined) {
        throw new Error('useAppContext must be used within AppProvider');
    }
    return context;
};

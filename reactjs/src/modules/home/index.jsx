import React, { useEffect } from 'react';

import useGlobalStore from '../../hook/useGlobal';

const HomeScreen = () => {
  const { test, check } = useGlobalStore();
  useEffect(() => {
    test('thu cu nha');
  });
  console.log('check str', check);
  return <div>Home page ccasdasdasd thai moi code</div>;
};

export default HomeScreen;